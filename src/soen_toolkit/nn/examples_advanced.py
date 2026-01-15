"""Advanced examples showing fine-grained parameter control.

These examples demonstrate how to use distributions, constraints, and learnability
for layer parameters.
"""

import torch

from soen_toolkit.nn import Graph, init, layers, param_specs, structure


def example_distributions() -> Graph:
    """Example: Using distributions for layer parameters."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))

    # Use distributions for SingleDendrite parameters
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=50,
            solver="FE",
            source_func_type="RateArray",
            # Uniform distribution for bias_current
            bias_current=param_specs.uniform(
                min=1.5,
                max=2.0,
                learnable=True,
                constraints={"min": 0.0, "max": 5.0},
            ),
            # Log-normal distribution for gamma_plus
            gamma_plus=param_specs.lognormal(
                mean=-6.9,
                std=0.2,  # exp(-6.9) ≈ 1e-3
                learnable=True,
                constraints={"min": 0.0, "max": 0.01},
            ),
            # Normal distribution for gamma_minus
            gamma_minus=param_specs.normal(
                mean=0.001,
                std=0.0002,
                learnable=False,  # Frozen
            ),
            # Constant phi_offset
            phi_offset=0.23,
        ),
    )

    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    x = torch.randn(2, 100, 10)
    g(x)

    # Check which parameters are trainable
    for _name, _param in g.named_parameters():
        pass

    return g


def example_learnability_control() -> Graph:
    """Example: Fine-grained control over parameter learnability."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))

    # Create layer with specific learnability settings
    layer_spec = layers.SingleDendrite(
        dim=20,
        solver="FE",
        source_func_type="RateArray",
        bias_current=1.7,
        gamma_plus=0.001,
        gamma_minus=0.001,
        phi_offset=0.23,
    )

    # Control learnability via add_layer
    g.add_layer(
        1,
        layer_spec,
        learnable_params={
            "bias_current": True,  # Train bias_current
            "gamma_plus": True,  # Train gamma_plus
            "gamma_minus": False,  # Freeze gamma_minus
            "phi_offset": False,  # Freeze phi_offset
        },
    )

    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    # Compile and check
    g.compile()

    trainable = []
    frozen = []
    for name, param in g.named_parameters():
        if param.requires_grad:
            trainable.append(name)
        else:
            frozen.append(name)

    return g


def example_constraints() -> Graph:
    """Example: Parameter constraints during training."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))

    # Use ParamSpec to set constraints
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=20,
            solver="FE",
            source_func_type="RateArray",
            bias_current=param_specs.uniform(
                min=1.5,
                max=2.0,
                learnable=True,
                constraints={"min": 0.0, "max": 5.0},  # Enforced during training
            ),
            gamma_plus=param_specs.constant(
                value=0.001,
                learnable=True,
                constraints={"min": 0.0, "max": 0.01},  # Keep positive and bounded
            ),
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
    )

    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    # After training steps, constraints are enforced automatically
    # by the core model's enforce_param_constraints() method

    return g


def example_heterogeneous_layer() -> Graph:
    """Example: Layer with different initialization per node type."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))

    # Create layer where each node gets different bias_current
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=20,
            solver="FE",
            source_func_type="RateArray",
            # Linearly spaced bias currents across nodes
            bias_current=param_specs.linear(
                min=1.5,
                max=2.0,
                learnable=True,
            ),
            # Exponentially spaced gamma_plus
            gamma_plus=param_specs.loglinear(
                min=-8.0,
                max=-6.0,  # exp(-8) to exp(-6)
                learnable=True,
                constraints={"min": 0.0, "max": 0.01},
            ),
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
    )

    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    return g


def example_combined() -> tuple[Graph, torch.optim.Adam]:
    """Example: Combining all features."""
    g = Graph(dt=37, network_evaluation_method="layerwise", track_phi=True)

    g.add_layer(0, layers.Linear(dim=10))

    # First hidden layer with distributions
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=50,
            solver="FE",
            source_func_type="RateArray",
            bias_current=param_specs.uniform(1.5, 2.0, learnable=True, constraints={"min": 0.0, "max": 5.0}),
            gamma_plus=param_specs.lognormal(-6.9, 0.2, learnable=True, constraints={"min": 0.0, "max": 0.01}),
            gamma_minus=param_specs.constant(0.001, learnable=False),
            phi_offset=0.23,
        ),
        learnable_params={
            "bias_current": True,
            "gamma_plus": True,
            "gamma_minus": False,  # Redundant with ParamSpec but shows override
            "phi_offset": False,
        },
    )

    # Second hidden layer with different settings
    g.add_layer(
        2,
        layers.SingleDendrite(
            dim=20,
            solver="FE",
            source_func_type="RateArray",
            bias_current=param_specs.normal(1.7, 0.1, learnable=True),
            gamma_plus=0.001,  # Scalar = constant for all
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
    )

    # Output layer
    g.add_layer(3, layers.NonLinear(dim=5, source_func_type="Tanh"))

    # Connections with different patterns
    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())
    g.connect(1, 2, structure=structure.sparse(0.5), init=init.uniform(-0.2, 0.2))
    g.connect(2, 3, structure=structure.dense(), init=init.xavier_uniform())

    # Run forward pass
    x = torch.randn(2, 100, 10)
    g(x)

    # Setup optimizer (only trainable parameters)
    optimizer = torch.optim.Adam(g.parameters(), lr=1e-3)

    return g, optimizer


def example_old_dict_syntax() -> Graph:
    """Example: Using the old dict syntax (still supported)."""
    g = Graph(dt=37, network_evaluation_method="layerwise")

    g.add_layer(0, layers.Linear(dim=10))

    # You can also use raw dict syntax
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=20,
            solver="FE",
            source_func_type="RateArray",
            bias_current={
                "distribution": "uniform",
                "params": {"min": 1.5, "max": 2.0},
                "learnable": True,
                "constraints": {"min": 0.0, "max": 5.0},
            },
            gamma_plus={
                "distribution": "lognormal",
                "params": {"mean": -6.9, "std": 0.2},
            },
            gamma_minus=0.001,
            phi_offset=0.23,
        ),
    )

    g.connect(0, 1, structure=structure.dense(), init=init.xavier_uniform())

    return g


if __name__ == "__main__":
    g1 = example_distributions()

    g2 = example_learnability_control()

    g3 = example_constraints()

    g4 = example_heterogeneous_layer()

    g5, opt5 = example_combined()

    g6 = example_old_dict_syntax()


def example_dynamic_v2_connections() -> Graph:
    """Example: Using dynamic v2 connections with multiplier v2 circuits.

    Dynamic v2 connections use a multiplier circuit with dual SQUID states
    and aggregated output. This design supports hardware without collection coils.
    """
    g = Graph(dt=37, network_evaluation_method="layerwise")

    # Input layer
    g.add_layer(0, layers.Linear(dim=10))

    # Hidden layer with v2 dynamic connections
    g.add_layer(
        1,
        layers.SingleDendrite(
            dim=50,
            solver="FE",
            source_func_type="RateArray",
            bias_current=1.7,
            gamma_plus=1e-3,
            gamma_minus=1e-3,
        ),
    )

    # Output layer
    g.add_layer(
        2,
        layers.DendriteReadout(
            dim=5,
            source_func_type="RateArray",
            bias_current=1.7,
        ),
    )

    # Connect 0->1 with NOCC (multiplier v2 circuit)
    g.connect(
        0,
        1,
        structure=structure.dense(),
        init=init.normal(mean=0.0, std=0.1),
        mode="NOCC",
        connection_params={
            "source_func": "RateArray",
            "alpha": 1.64053,  # Dimensionless resistance
            "beta": 303.85,  # Inductance of incoming branches
            "beta_out": 91.156,  # Inductance of output branch
            "bias_current": 2.1,  # Bias current
        },
    )

    # Connect 1->2 with WICC (multiplier v1 circuit)
    g.connect(
        1,
        2,
        structure=structure.dense(),
        init=init.normal(mean=0.0, std=0.1),
        mode="WICC",
        connection_params={
            "source_func": "RateArray",
            "gamma_plus": 1e-3,
            "bias_current": 2.0,
        },
    )

    # Internal recurrent connection with NOCC dynamics
    g.connect(
        1,
        1,
        structure=structure.sparse(sparsity=0.3),
        init=init.uniform(-0.1, 0.1),
        mode="NOCC",
        connection_params={
            "source_func": "RateArray",
            "alpha": 1.64053,
            "beta": 303.85,
            "beta_out": 91.156,
            "bias_current": 2.1,
        },
        allow_self_connections=False,
    )

    return g


if __name__ == "__main__":
    g1 = example_distributions()

    g2 = example_learnability_control()

    g3 = example_constraints()

    g4 = example_heterogeneous_layer()

    g5, opt5 = example_combined()

    g6 = example_old_dict_syntax()

    g7 = example_dynamic_v2_connections()
