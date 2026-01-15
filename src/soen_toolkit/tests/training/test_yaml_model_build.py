from pathlib import Path

import torch

from soen_toolkit.core.model_yaml import build_model_from_yaml
from soen_toolkit.training.configs.config_classes import (
    DataConfig,
    ExperimentConfig,
    LoggingConfig,
    ModelConfig,
    TrainingConfig,
)
from soen_toolkit.training.models.lightning_wrapper import SOENLightningModule


def _example_yaml_path() -> Path:
    return Path(__file__).resolve().parents[4] / "src" / "soen_toolkit" / "training" / "examples" / "training_configs" / "model_architecture.example.yaml"


def test_build_core_model_from_yaml_and_forward() -> None:
    path = _example_yaml_path()
    model = build_model_from_yaml(path)
    assert len(model.layers_config) == 3
    assert any(k.startswith("J_") for k in model.connections)

    # Run a tiny forward
    batch, seq_len, input_dim = 2, 5, model.layers_config[0].params["dim"]
    x = torch.randn(batch, seq_len, input_dim)
    y, all_states = model(x)
    assert y.shape[0] == batch
    assert len(all_states) == len(model.layers_config)


def test_constraints_and_learnability_respected() -> None:
    path = _example_yaml_path()
    model = build_model_from_yaml(path)
    # Apply constraints by a forward (model enforces pre-forward)
    input_dim = model.layers_config[0].params["dim"]
    _ = model(torch.randn(1, 2, input_dim))
    for name, param in model.connections.items():
        cons = model.connection_constraints.get(name, {})
        if cons:
            if "min" in cons:
                assert torch.all(param >= cons["min"]) or torch.allclose(param.min(), torch.tensor(cons["min"]).to(param.device), atol=1e-6)
            if "max" in cons:
                assert torch.all(param <= cons["max"]) or torch.allclose(param.max(), torch.tensor(cons["max"]).to(param.device), atol=1e-6)


def test_lightning_module_builds_from_yaml(tmp_path) -> None:
    # Minimal training config; data fields unused in this unit test
    cfg = ExperimentConfig(
        training=TrainingConfig(max_epochs=1, batch_size=2),
        data=DataConfig(data_path=str(tmp_path), sequence_length=5, num_classes=10),
        model=ModelConfig(architecture_yaml=str(_example_yaml_path())),
        logging=LoggingConfig(project_dir=str(tmp_path)),
    )
    lm = SOENLightningModule(cfg)
    assert lm.model is not None
    assert len(lm.model.layers_config) == 3
