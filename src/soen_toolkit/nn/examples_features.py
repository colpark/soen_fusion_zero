"""Examples demonstrating all SOENModelCore features in the PyTorch API.

This script shows that Graph and Sequential have full access to:
- Visualization
- Summary and statistics
- Save/Load (.soen, .pth, .json formats)
- All other SOENModelCore features
"""

import contextlib

import torch

from soen_toolkit.nn import Graph, Sequential, init, layers, structure


def example_visualization() -> Graph:
    """Example: Network visualization."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=50,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=0.001,
            gamma_minus=0.001,
        ),
    )
    g.add_layer(2, layers.NonLinear(dim=5, source_func_type="Tanh"))

    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
    g.connect(1, 2, structure=structure.dense(), init=init.xavier_uniform())

    # Visualize the network
    g.visualize(
        save_path="example_network",
        file_format="png",
        orientation="LR",
        simple_view=True,
    )

    return g


def example_summary() -> Sequential:
    """Example: Model summary and statistics."""
    # Build a sequential model
    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=50,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=0.001,
                gamma_minus=0.001,
            ),
            layers.NonLinear(dim=5, source_func_type="Tanh"),
        ]
    )

    # Print summary
    net.summary(print_summary=True, verbose=False)

    # Get summary as dict
    stats = net.compute_summary()
    stats.get("total_parameters", 0)
    stats.get("trainable_parameters", 0)

    # Get summary as DataFrame
    net.summary(return_df=True, print_summary=False)

    return net


def example_save_load() -> tuple[Graph, Graph]:
    """Example: Save and load models."""
    # Build a model
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=20,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=0.001,
            gamma_minus=0.001,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    # Compile and run forward to initialize
    x = torch.randn(2, 10, 10)
    output1 = g(x)

    # Save in different formats
    g.save("test_model.soen")

    g.save("test_model.pth")

    g.save("test_model.json")

    # Load and verify
    g_loaded = Graph.load("test_model.soen")

    # Verify loaded model produces same output
    output2 = g_loaded(x)

    # Check if outputs match
    if torch.allclose(output1, output2, atol=1e-6):
        pass
    else:
        pass

    # Clean up test files
    import os

    for ext in [".soen", ".pth", ".json"]:
        with contextlib.suppress(Exception):
            os.remove(f"test_model{ext}")

    return g, g_loaded


def example_state_dict() -> tuple[Graph, Graph]:
    """Example: Working with state dictionaries."""

    # Build two identical models
    def build_model() -> Graph:
        g = Graph(dt=37, network_evaluation_method="layerwise")
        g.add_layer(0, layers.Linear(dim=10))
        g.add_layer(
            1,
            layers.SingleDendrite(
                dim=20,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=0.001,
                gamma_minus=0.001,
            ),
        )
        g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
        return g

    model1 = build_model()
    model2 = build_model()

    # Initialize with random forward pass
    x = torch.randn(2, 10, 10)
    model1(x)
    model2(x)

    # Transfer weights from model1 to model2
    state = model1.state_dict()
    model2.load_state_dict(state)

    # Verify they now produce same output
    out1_after = model1(x)
    out2_after = model2(x)

    if torch.allclose(out1_after, out2_after, atol=1e-6):
        pass

    return model1, model2


def example_device_management() -> Graph:
    """Example: Moving models between devices."""
    g = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=20,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=0.001,
                gamma_minus=0.001,
            ),
        ]
    )

    # Check available devices
    has_mps = torch.backends.mps.is_available()
    has_cuda = torch.cuda.is_available()

    # Test on CPU
    x_cpu = torch.randn(2, 10, 10)
    g(x_cpu)

    # Test on GPU if available
    if has_cuda:
        g_cuda = g.to("cuda")
        x_cuda = x_cpu.to("cuda")
        g_cuda(x_cuda)
        g = g.to("cpu")  # type: ignore[assignment]  # Move back to CPU

    if has_mps:
        g_mps = g.to("mps")
        x_mps = x_cpu.to("mps")
        g_mps(x_mps)

    return g


def example_training_with_checkpoints() -> Sequential:
    """Example: Training with periodic checkpoints."""
    # Build model
    net = Sequential(
        [
            layers.Linear(dim=10),
            layers.SingleDendrite(
                dim=20,
                solver="FE",
                source_func_type="RateArray",
                bias_current=1.7,
                gamma_plus=0.001,
                gamma_minus=0.001,
            ),
            layers.NonLinear(dim=5, source_func_type="Tanh"),
        ]
    )

    # Dummy data
    x_train = torch.randn(16, 100, 10)
    y_train = torch.randint(0, 5, (16,))

    # Compile to ensure parameters are registered
    net.compile()

    # Setup
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    # Training loop with checkpoints
    for epoch in range(5):
        optimizer.zero_grad()
        output = net(x_train)
        loss = criterion(output[:, -1, :], y_train)
        loss.backward()
        optimizer.step()

        # Save checkpoint every 2 epochs
        if (epoch + 1) % 2 == 0:
            checkpoint_path = f"checkpoint_epoch_{epoch + 1}.soen"
            net.save(checkpoint_path)

    # Clean up checkpoints
    import os

    for epoch in [2, 4]:
        with contextlib.suppress(Exception):
            os.remove(f"checkpoint_epoch_{epoch}.soen")

    return net


def example_accessing_core() -> Graph:
    """Example: Accessing the underlying SOENModelCore for advanced use."""
    g = Graph(dt=37, network_evaluation_method="layerwise")
    g.add_layer(0, layers.Linear(dim=10))
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=20,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=0.001,
            gamma_minus=0.001,
        ),
    )
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    # Compile to create core
    g.compile()

    # Access the underlying core
    core = g._compiled_core
    assert core is not None

    # Access core-specific attributes

    # Direct access to layers and connections
    for _i, _layer in enumerate(core.layers):
        pass

    for _name, _weight in core.connections.items():
        pass

    return g


if __name__ == "__main__":
    # Run all examples
    g1 = example_visualization()
    net1 = example_summary()
    g2, g2_loaded = example_save_load()
    m1, m2 = example_state_dict()
    g3 = example_device_management()
    net2 = example_training_with_checkpoints()
    g4 = example_accessing_core()
