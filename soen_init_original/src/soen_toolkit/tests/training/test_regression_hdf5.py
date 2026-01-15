import pytorch_lightning as pl
import torch

from soen_toolkit.tests.utils.hdf5_test_helpers import make_temp_hdf5_regression
from soen_toolkit.tests.utils.test_helpers_fixture import build_small_model
from soen_toolkit.training.configs.config_classes import (
    DataConfig,
    ExperimentConfig,
    LossConfig,
    LossItemConfig,
    ModelConfig,
    TrainingConfig,
)
from soen_toolkit.training.data.data_module import SOENDataModule
from soen_toolkit.training.models.lightning_wrapper import SOENLightningModule


class DummyWrapper(SOENLightningModule):
    def __init__(self, config: ExperimentConfig, core_model=None, input_dim=6, output_dim=3, pooling="final") -> None:
        pl.LightningModule.__init__(self)
        self.config = config
        self.model = core_model if core_model is not None else build_small_model(dims=(input_dim, output_dim))
        # parse pooling
        tp_cfg = self.config.model.time_pooling
        if isinstance(tp_cfg, dict):
            self.time_pooling_method_name = tp_cfg.get("name", pooling)
            self.time_pooling_params = tp_cfg.get("params", {})
        else:
            self.time_pooling_method_name = tp_cfg
            self.time_pooling_params = {}
        if "scale" not in self.time_pooling_params:
            self.time_pooling_params["scale"] = 1.0
        self.range_start = None
        self.range_end = None
        self.autoregressive = False
        self.latest_processed_state = None
        self.latest_final_state = None
        self.latest_all_states = None
        self._initialize_loss_functions()


def _make_config(h5_path: str, *, mapping: str, target_seq_len: int, pooling: dict | str | None = None, losses=None) -> ExperimentConfig:
    if pooling is None:
        pooling = {"name": "final", "params": {"scale": 1.0}}
    if losses is None:
        losses = [LossItemConfig(name="mse", weight=1.0, params={})]
    loss_cfg = LossConfig(losses=losses)
    train_cfg = TrainingConfig(batch_size=8, max_epochs=1, loss=loss_cfg, paradigm="supervised", mapping=mapping)
    data_cfg = DataConfig(
        data_path=h5_path,
        cache_data=True,
        num_classes=1,
        val_split=0.2,
        test_split=0.1,
        target_seq_len=target_seq_len,
    )
    model_cfg = ModelConfig(base_model_path=None, time_pooling=pooling)
    return ExperimentConfig(training=train_cfg, data=data_cfg, model=model_cfg)


def test_regression_seq2static_mse_trains(tmp_path) -> None:
    # Static float targets: shape [N, target_dim]
    seq_len = 20
    with make_temp_hdf5_regression(num_samples=48, seq_len=seq_len, feat_dim=6, target_dim=3, target_sequence=False, with_splits=True) as (h5_path, tdim):
        cfg = _make_config(h5_path, mapping="seq2static", target_seq_len=seq_len, pooling={"name": "mean", "params": {"scale": 1.0}})
        dm = SOENDataModule(cfg)
        dm.prepare_data()
        dm.setup()
        x, y = next(iter(dm.train_dataloader()))
        assert y.dtype.is_floating_point
        wrapper = DummyWrapper(cfg, core_model=build_small_model(dims=(6, tdim)), input_dim=6, output_dim=tdim, pooling="mean")
        res = wrapper.training_step((x, y), batch_idx=0)
        loss = res["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
        assert any(g is not None for g in grads)


def test_regression_seq2seq_mse_trains(tmp_path) -> None:
    # Sequence float targets: shape [N, T, target_dim]
    seq_len = 12
    with make_temp_hdf5_regression(num_samples=40, seq_len=seq_len, feat_dim=5, target_dim=2, target_sequence=True, with_splits=True) as (h5_path, tdim):
        cfg = _make_config(h5_path, mapping="seq2seq", target_seq_len=seq_len, pooling={"name": "final", "params": {"scale": 1.0}})
        dm = SOENDataModule(cfg)
        dm.prepare_data()
        dm.setup()
        x, y = next(iter(dm.train_dataloader()))
        assert y.dtype.is_floating_point
        assert y.ndim == 3
        # Use sequence outputs: Lightning wrapper will pass full sequence when NOT autoregressive
        wrapper = DummyWrapper(cfg, core_model=build_small_model(dims=(5, tdim)), input_dim=5, output_dim=tdim, pooling="final")
        # Force supervised flow: y shouldn't be overridden
        res = wrapper.training_step((x, y), batch_idx=0)
        loss = res["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
        assert any(g is not None for g in grads)
