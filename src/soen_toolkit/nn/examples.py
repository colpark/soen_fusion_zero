"""Examples demonstrating the PyTorch-style API.

These examples show how to use Graph and Sequential for common model building tasks.
"""

import torch

from soen_toolkit.nn import Graph, Sequential, init, layers, structure


def example_simple_feedforward() -> tuple[Sequential, torch.optim.Adam]:
    """Simple feedforward network using Sequential."""
    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=50,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=1e-3,
                gamma_minus=1e-3,
            ),
            layers.DendriteReadout(
                dim=5,
                source_func_type="RateArray",
                bias_current=1.7,
            ),
        ]
    )

    # Standard PyTorch usage
    x = torch.randn(2, 100, 10)  # batch, time, features
    net(x)

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

    return net, optimizer


def example_graph_with_custom_connections() -> Graph:
    """Graph with custom connection structures."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=20,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.add_layer(
        2,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )

    # Dense connection with Xavier init
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform(gain=1.0))

    # Sparse connection with custom init
    g.connect(1, 2, structure=structure.sparse(sparsity=0.3), init=init.uniform(-0.2, 0.2))

    return g


def example_recurrent_with_feedback() -> Graph:
    """Recurrent network with feedback connections."""
    g = Graph(dt=37, network_evaluation_method="stepwise_gauss_seidel")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=20,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.add_layer(
        2,
        layers.SingleDendrite(
            dim=15,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )

    # Forward connections
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
    g.connect(1, 2, structure=structure.dense(), init=init.xavier_uniform())

    # Feedback connection
    g.connect(2, 1, structure=structure.sparse(0.3), init=init.uniform(-0.1, 0.1))

    return g


def example_dynamic_connections() -> Graph:
    """Network with dynamic (multiplier-based) connections."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )

    # Dynamic connection with multiplier circuits
    g.connect(
        0,
        1,
        structure=structure.sparse(sparsity=0.5),
        init=init.uniform(-0.15, 0.15),
        mode="WICC",
        connection_params={
            "source_func": "RateArray",
            "gamma_plus": 1e-3,
            "bias_current": 2.0,
        },
    )

    return g


def example_hybrid_network() -> Graph:
    """Hybrid network mixing SOEN and traditional RNN layers."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(1, layers.LSTM(dim=32))
    g.add_layer(
        2,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )

    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
    g.connect(1, 2, structure=structure.dense(), init=init.xavier_uniform())

    return g


def example_with_tracking() -> tuple[Graph, list[torch.Tensor] | None, list[torch.Tensor] | None, list[torch.Tensor] | None]:
    """Network with history tracking enabled."""
    g = Graph(
        dt=37,
        network_evaluation_method="layerwise",
        track_phi=True,
        track_s=True,
        track_power=True,
    )

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=5,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    x = torch.randn(2, 100, 10)
    g(x)

    # Access histories
    phi_history = g.phi_history  # List of tensors per layer
    s_history = g.s_history
    power_history = g.power_history

    return g, phi_history, s_history, power_history


def example_save_and_load() -> Sequential:
    """Save and load a model."""
    import os
    import tempfile

    # Build and train
    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=5,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=1e-3,
                gamma_minus=1e-3,
            ),
        ]
    )

    x = torch.randn(2, 10, 10)
    output = net(x)

    # Save
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "model.soen")
        net.save(path)

        # Load
        loaded_net = Sequential.load(path)
        output_loaded = loaded_net(x)

        assert torch.allclose(output, output_loaded)

    return net


def example_training_loop() -> Sequential:
    """Complete training loop example."""
    # Build model
    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=20,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=1e-3,
                gamma_minus=1e-3,
            ),
            layers.DendriteReadout(dim=5, source_func_type="RateArray", bias_current=1.7),
        ]
    )

    # Dummy data
    x_train = torch.randn(32, 100, 10)  # batch, time, features
    y_train = torch.randint(0, 5, (32,))  # batch of class labels

    # Compile model (or let first forward do it)
    net.compile()

    # Setup optimizer and loss
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop
    net.train()
    for _epoch in range(5):
        optimizer.zero_grad()

        # Forward pass
        output = net(x_train)  # (batch, time, classes)

        # Use final timestep for classification
        logits = output[:, -1, :]  # (batch, classes)

        # Compute loss and backprop
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    return net


if __name__ == "__main__":
    net1, opt1 = example_simple_feedforward()

    net2 = example_graph_with_custom_connections()

    net3 = example_recurrent_with_feedback()

    net4 = example_dynamic_connections()

    example_training_loop()
