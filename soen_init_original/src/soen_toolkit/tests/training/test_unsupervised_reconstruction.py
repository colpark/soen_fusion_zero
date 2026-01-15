import pytorch_lightning as pl
import torch

from soen_toolkit.tests.utils.test_helpers_fixture import (
    build_small_model,
    make_random_series,
)
from soen_toolkit.training.configs.config_classes import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    LossItemConfig,
    ModelConfig,
    TrainingConfig,
)
from soen_toolkit.training.models.lightning_wrapper import SOENLightningModule


class DummyWrapper(SOENLightningModule):
    def __init__(self, config: ExperimentConfig, core_model=None) -> None:
        # Initialize nn.Module/Lightning internals so assigning submodules works
        pl.LightningModule.__init__(self)
        # Bypass parent SOENLightningModule.__init__ to avoid filesystem loads
        self.config = config
        self.model = core_model if core_model is not None else build_small_model(dims=(3, 3))
        # Minimal attributes used by forward/process_output
        self.time_pooling_method_name = "final"
        self.time_pooling_params = {"scale": 1.0}
        self.range_start = None
        self.range_end = None
        self.autoregressive = False
        self.latest_processed_state = None
        self.latest_final_state = None
        self.latest_all_states = None
        self._initialize_loss_functions()


def _make_config(paradigm: str = "unsupervised") -> ExperimentConfig:
    loss_cfg = LossConfig(losses=[LossItemConfig(name="mse", weight=1.0, params={})])
    train_cfg = TrainingConfig(batch_size=2, max_epochs=1, loss=loss_cfg, paradigm=paradigm, mapping="seq2seq")
    data_cfg = DataConfig(sequence_length=6)
    # Provide a dummy base_model_path by bypassing in DummyWrapper
    model_cfg = ModelConfig(base_model_path=None)
    return ExperimentConfig(training=train_cfg, data=data_cfg, model=model_cfg)


def test_unsupervised_reconstruction_uses_inputs_as_targets_and_backprops() -> None:
    B, T, D = 2, 5, 3
    x = make_random_series(B, T, D, seed=42)
    # Provide dummy scalar labels to simulate missing targets
    y = torch.zeros(B, dtype=torch.long)

    cfg = _make_config(paradigm="unsupervised")
    wrapper = DummyWrapper(cfg, core_model=build_small_model(dims=(D, D)))

    # Forward once to set caches; then call training_step
    out, _, _ = wrapper(x)
    assert out.shape[0] == B

    res = wrapper.training_step((x, y), batch_idx=0)
    loss = res["loss"]
    assert torch.isfinite(loss), "Loss should be finite"
    loss.backward()
    # Ensure some gradients exist on connections
    grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads), "At least one parameter should receive gradients"


def test_unsupervised_reconstruction_tbptt_chunks_backprop() -> None:
    # Configure TBPTT
    B, T, D = 2, 11, 4
    x = make_random_series(B, T, D, seed=7)
    y = torch.zeros(B, dtype=torch.long)

    loss_cfg = LossConfig(losses=[LossItemConfig(name="mse", weight=1.0, params={})])
    train_cfg = TrainingConfig(
        batch_size=2,
        max_epochs=1,
        loss=loss_cfg,
        paradigm="unsupervised",
        mapping="seq2seq",
        use_tbptt=True,
        tbptt_steps=4,
        tbptt_stride=4,
    )
    cfg = ExperimentConfig(training=train_cfg, data=DataConfig(sequence_length=T), model=ModelConfig(base_model_path=None))
    wrapper = DummyWrapper(cfg, core_model=build_small_model(dims=(D, D)))

    res = wrapper.training_step((x, y), batch_idx=0)
    loss = res["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)
