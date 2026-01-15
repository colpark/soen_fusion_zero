import pytorch_lightning as pl
import torch

from soen_toolkit.tests.utils.hdf5_test_helpers import make_temp_hdf5_classification
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


def _make_config(h5_path: str, num_classes: int, *, pooling: dict | str | None = None, losses=None) -> ExperimentConfig:
    if pooling is None:
        pooling = {"name": "final", "params": {"scale": 1.0}}
    if losses is None:
        losses = [LossItemConfig(name="cross_entropy", weight=1.0, params={})]
    loss_cfg = LossConfig(losses=losses)
    train_cfg = TrainingConfig(batch_size=8, max_epochs=1, loss=loss_cfg, paradigm="supervised", mapping="seq2static")
    data_cfg = DataConfig(
        data_path=h5_path,
        cache_data=True,
        num_classes=num_classes,
        val_split=0.2,
        test_split=0.1,
        target_seq_len=20,
    )
    model_cfg = ModelConfig(base_model_path=None, time_pooling=pooling)
    return ExperimentConfig(training=train_cfg, data=data_cfg, model=model_cfg)


class DummyWrapper(SOENLightningModule):
    def __init__(self, config: ExperimentConfig, core_model=None, input_dim=8, num_classes=5) -> None:
        pl.LightningModule.__init__(self)
        self.config = config
        self.model = core_model if core_model is not None else build_small_model(dims=(input_dim, num_classes))
        # parse pooling
        tp_cfg = self.config.model.time_pooling
        if isinstance(tp_cfg, dict):
            self.time_pooling_method_name = tp_cfg.get("name", "final")
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


def test_classification_hdf5_one_batch_ce_trains_and_backprops(tmp_path) -> None:
    with make_temp_hdf5_classification(num_samples=50, seq_len=20, feat_dim=8, num_classes=5, with_splits=True) as (h5_path, n_classes):
        cfg = _make_config(h5_path, n_classes)
        # Build dataloaders directly (not full Trainer)
        dm = SOENDataModule(cfg)
        dm.prepare_data()
        dm.setup()
        train_loader = dm.train_dataloader()

        wrapper = DummyWrapper(cfg, core_model=build_small_model(dims=(8, n_classes)), input_dim=8, num_classes=n_classes)

        x, y = next(iter(train_loader))
        out = wrapper.forward(x)[0]
        assert out.shape[0] == x.shape[0]

        res = wrapper.training_step((x, y), batch_idx=0)
        loss = res["loss"]
        assert torch.isfinite(loss)
        loss.backward()
        grads = [p.grad for p in wrapper.model.parameters() if p.requires_grad]
        assert any(g is not None for g in grads)


def test_classification_pooling_variants_run(tmp_path) -> None:
    with make_temp_hdf5_classification(num_samples=30, seq_len=12, feat_dim=6, num_classes=3, with_splits=False) as (h5_path, n_classes):
        for pooling in (
            {"name": "final", "params": {"scale": 1.0}},
            {"name": "mean", "params": {"scale": 1.0}},
            {"name": "mean_last_n", "params": {"n": 3, "scale": 1.0}},
        ):
            cfg = _make_config(h5_path, n_classes, pooling=pooling)
            dm = SOENDataModule(cfg)
            dm.prepare_data()
            dm.setup()
            x, y = next(iter(dm.train_dataloader()))
            wrapper = DummyWrapper(cfg, core_model=build_small_model(dims=(6, n_classes)), input_dim=6, num_classes=n_classes)
            res = wrapper.training_step((x, y), batch_idx=0)
            assert torch.isfinite(res["loss"])


def test_classification_weighted_losses_aggregate(tmp_path) -> None:
    with make_temp_hdf5_classification(num_samples=40, seq_len=16, feat_dim=7, num_classes=4, with_splits=True) as (h5_path, n_classes):
        losses = [
            LossItemConfig(name="cross_entropy", weight=1.0, params={}),
            LossItemConfig(name="gap_loss", weight=0.5, params={"margin": 0.2}),
        ]
        cfg = _make_config(h5_path, n_classes, losses=losses)
        dm = SOENDataModule(cfg)
        dm.prepare_data()
        dm.setup()
        x, y = next(iter(dm.train_dataloader()))
        wrapper = DummyWrapper(cfg, core_model=build_small_model(dims=(7, n_classes)), input_dim=7, num_classes=n_classes)
        out = wrapper.training_step((x, y), batch_idx=0)
        assert torch.isfinite(out["loss"])
