"""Simple test models for training pipeline tests.
Creates standardized models for each task type.
"""

from soen_toolkit.core import (
    ConnectionConfig,
    LayerConfig,
    SimulationConfig,
    SOENModelCore,
)


def build_classification_model(input_dim: int = 6, hidden_dim: int = 8, num_classes: int = 3) -> SOENModelCore:
    """Build a simple 3-layer model for classification tasks."""
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": input_dim}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": hidden_dim}),
        LayerConfig(layer_id=2, layer_type="RNN", params={"dim": num_classes}),
    ]

    connections = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "xavier_uniform", "learnable": True},
            learnable=True,
        ),
        ConnectionConfig(
            from_layer=1,
            to_layer=2,
            connection_type="dense",
            params={"init": "xavier_uniform", "learnable": True},
            learnable=True,
        ),
    ]

    sim = SimulationConfig(
        dt=1.0,
        input_type="flux",
        track_phi=False,
        track_s=False,
        track_g=False,
    )

    return SOENModelCore(
        sim_config=sim,
        layers_config=layers,
        connections_config=connections,
    )


def build_regression_model(input_dim: int = 6, hidden_dim: int = 10, output_dim: int = 3) -> SOENModelCore:
    """Build a simple 3-layer model for regression tasks."""
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": input_dim}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": hidden_dim}),
        LayerConfig(layer_id=2, layer_type="RNN", params={"dim": output_dim}),
    ]

    connections = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "xavier_normal", "learnable": True},
            learnable=True,
        ),
        ConnectionConfig(
            from_layer=1,
            to_layer=2,
            connection_type="dense",
            params={"init": "xavier_normal", "learnable": True},
            learnable=True,
        ),
    ]

    sim = SimulationConfig(
        dt=1.0,
        input_type="flux",
        track_phi=False,
        track_s=False,
        track_g=False,
    )

    return SOENModelCore(
        sim_config=sim,
        layers_config=layers,
        connections_config=connections,
    )


def build_sequence_model(input_dim: int = 6, hidden_dim: int = 12, output_dim: int = 6) -> SOENModelCore:
    """Build a model with recurrent connections for sequence-to-sequence tasks."""
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": input_dim}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": hidden_dim}),
        LayerConfig(layer_id=2, layer_type="RNN", params={"dim": output_dim}),
    ]

    connections = [
        # Feedforward connections
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "xavier_uniform", "learnable": True},
            learnable=True,
        ),
        ConnectionConfig(
            from_layer=1,
            to_layer=2,
            connection_type="dense",
            params={"init": "xavier_uniform", "learnable": True},
            learnable=True,
        ),
        # Add recurrent connection in hidden layer for temporal dynamics
        ConnectionConfig(
            from_layer=1,
            to_layer=1,
            connection_type="dense",
            params={"init": "orthogonal", "learnable": True},
            learnable=True,
        ),
    ]

    sim = SimulationConfig(
        dt=1.0,
        input_type="flux",
        track_phi=False,
        track_s=False,
        track_g=False,
    )

    return SOENModelCore(
        sim_config=sim,
        layers_config=layers,
        connections_config=connections,
    )


def build_autoencoder_model(input_dim: int = 6, hidden_dim: int = 4) -> SOENModelCore:
    """Build a simple autoencoder model for unsupervised learning."""
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": input_dim}),  # Input
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": hidden_dim}),  # Bottleneck
        LayerConfig(layer_id=2, layer_type="RNN", params={"dim": input_dim}),  # Reconstruction
    ]

    connections = [
        # Encoder
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "xavier_uniform", "learnable": True},
            learnable=True,
        ),
        # Decoder
        ConnectionConfig(
            from_layer=1,
            to_layer=2,
            connection_type="dense",
            params={"init": "xavier_uniform", "learnable": True},
            learnable=True,
        ),
    ]

    sim = SimulationConfig(
        dt=1.0,
        input_type="flux",
        track_phi=False,
        track_s=False,
        track_g=False,
    )

    return SOENModelCore(
        sim_config=sim,
        layers_config=layers,
        connections_config=connections,
    )


def build_pulse_classification_model(input_dim: int = 1, hidden_dim: int = 5, num_classes: int = 3) -> SOENModelCore:
    """Build a specialized model for pulse classification similar to the tutorial."""
    layers = [
        LayerConfig(layer_id=0, layer_type="RNN", params={"dim": input_dim}),
        LayerConfig(layer_id=1, layer_type="RNN", params={"dim": hidden_dim}),
        LayerConfig(layer_id=2, layer_type="RNN", params={"dim": num_classes}),
    ]

    connections = [
        ConnectionConfig(
            from_layer=0,
            to_layer=1,
            connection_type="dense",
            params={"init": "xavier_uniform", "learnable": True},
            learnable=True,
        ),
        ConnectionConfig(
            from_layer=1,
            to_layer=2,
            connection_type="dense",
            params={"init": "xavier_uniform", "learnable": True},
            learnable=True,
        ),
        # Add some recurrence for temporal pattern detection
        ConnectionConfig(
            from_layer=1,
            to_layer=1,
            connection_type="dense",
            params={"init": "orthogonal", "learnable": True},
            learnable=True,
        ),
    ]

    sim = SimulationConfig(
        dt=1.0,
        input_type="flux",
        track_phi=False,
        track_s=False,
        track_g=False,
    )

    return SOENModelCore(
        sim_config=sim,
        layers_config=layers,
        connections_config=connections,
    )
