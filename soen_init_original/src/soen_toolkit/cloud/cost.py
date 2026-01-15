"""Cost estimation for cloud jobs.

Provides cost estimates before job submission using AWS instance pricing.
Includes both on-demand and spot pricing estimates.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import InstanceConfig, JobConfig

logger = logging.getLogger(__name__)


# SageMaker instance pricing (US East 1, as of 2024)
# Source: https://aws.amazon.com/sagemaker/pricing/
# These are fallback values - ideally fetch from AWS Pricing API
INSTANCE_PRICES_USD_PER_HOUR: dict[str, float] = {
    # G4 instances (T4 GPU)
    "ml.g4dn.xlarge": 0.736,
    "ml.g4dn.2xlarge": 1.053,
    "ml.g4dn.4xlarge": 1.686,
    "ml.g4dn.8xlarge": 3.045,
    "ml.g4dn.12xlarge": 5.477,
    "ml.g4dn.16xlarge": 6.090,
    # G5 instances (A10G GPU)
    "ml.g5.xlarge": 1.006,
    "ml.g5.2xlarge": 1.515,
    "ml.g5.4xlarge": 2.534,
    "ml.g5.8xlarge": 4.571,
    "ml.g5.12xlarge": 7.094,
    "ml.g5.16xlarge": 8.142,
    "ml.g5.24xlarge": 11.287,
    "ml.g5.48xlarge": 20.574,
    # P3 instances (V100 GPU)
    "ml.p3.2xlarge": 3.825,
    "ml.p3.8xlarge": 14.688,
    "ml.p3.16xlarge": 28.152,
    # P4 instances (A100 GPU)
    "ml.p4d.24xlarge": 32.773,
    "ml.p4de.24xlarge": 40.966,
    # P5 instances (H100 GPU)
    "ml.p5.48xlarge": 98.32,
    # M5 instances (CPU only)
    "ml.m5.large": 0.115,
    "ml.m5.xlarge": 0.230,
    "ml.m5.2xlarge": 0.461,
    "ml.m5.4xlarge": 0.922,
    "ml.m5.12xlarge": 2.765,
    "ml.m5.24xlarge": 5.530,
    # C5 instances (CPU only, compute optimized)
    "ml.c5.large": 0.102,
    "ml.c5.xlarge": 0.204,
    "ml.c5.2xlarge": 0.408,
    "ml.c5.4xlarge": 0.816,
    "ml.c5.9xlarge": 1.836,
    "ml.c5.18xlarge": 3.672,
}

# Spot pricing is typically 50-70% discount
SPOT_DISCOUNT_FACTOR = 0.35  # Assume 65% discount


@dataclass(frozen=True)
class CostEstimate:
    """Cost estimate for a cloud job."""

    instance_type: str
    instance_count: int
    max_runtime_hours: float
    on_demand_hourly: float
    spot_hourly: float
    on_demand_total: float
    spot_total: float
    use_spot: bool

    @property
    def estimated_cost(self) -> float:
        """Get the estimated cost based on spot/on-demand setting."""
        return self.spot_total if self.use_spot else self.on_demand_total

    def format(self) -> str:
        """Format cost estimate as human-readable string."""
        lines = [
            "Cost Estimate:",
            f"  Instance: {self.instance_type} x {self.instance_count}",
            f"  Max Runtime: {self.max_runtime_hours:.1f} hours",
            f"  On-Demand: ${self.on_demand_hourly:.3f}/hr = ${self.on_demand_total:.2f} max",
            f"  Spot (~65% off): ${self.spot_hourly:.3f}/hr = ${self.spot_total:.2f} max",
        ]
        if self.use_spot:
            lines.append(f"  Using: Spot instances (estimated ${self.spot_total:.2f})")
        else:
            lines.append(f"  Using: On-demand (estimated ${self.on_demand_total:.2f})")
        return "\n".join(lines)


class CostEstimator:
    """Estimates costs for cloud jobs.

    Uses cached pricing data with optional AWS Pricing API lookup.
    """

    def __init__(self, region: str = "us-east-1") -> None:
        """Initialize cost estimator.

        Args:
            region: AWS region for pricing lookup.
        """
        self.region = region
        self._price_cache = dict(INSTANCE_PRICES_USD_PER_HOUR)

    def get_hourly_price(self, instance_type: str) -> float:
        """Get hourly on-demand price for an instance type.

        Args:
            instance_type: SageMaker instance type (e.g., ml.g5.xlarge).

        Returns:
            Hourly price in USD.

        Raises:
            ValueError: If instance type is unknown.
        """
        if instance_type in self._price_cache:
            return self._price_cache[instance_type]

        # Try to infer from similar instance
        base = instance_type.rsplit(".", 1)[0] if "." in instance_type else instance_type
        for known_type, price in self._price_cache.items():
            if known_type.startswith(base):
                logger.warning(
                    f"Unknown instance type '{instance_type}', "
                    f"estimating from '{known_type}': ${price:.3f}/hr"
                )
                return price

        raise ValueError(
            f"Unknown instance type: {instance_type}. "
            f"Known types: {', '.join(sorted(self._price_cache.keys()))}"
        )

    def estimate(self, config: InstanceConfig | JobConfig) -> CostEstimate:
        """Estimate cost for a job configuration.

        Args:
            config: Instance or job configuration.

        Returns:
            CostEstimate with on-demand and spot pricing.
        """
        # Handle both InstanceConfig and JobConfig
        from .config import JobConfig

        if isinstance(config, JobConfig):
            instance_config = config.instance
        else:
            instance_config = config

        instance_type = instance_config.instance_type
        instance_count = instance_config.instance_count
        max_hours = instance_config.max_runtime_hours
        use_spot = instance_config.use_spot

        hourly_price = self.get_hourly_price(instance_type)

        on_demand_hourly = hourly_price * instance_count
        spot_hourly = on_demand_hourly * SPOT_DISCOUNT_FACTOR

        on_demand_total = on_demand_hourly * max_hours
        spot_total = spot_hourly * max_hours

        return CostEstimate(
            instance_type=instance_type,
            instance_count=instance_count,
            max_runtime_hours=max_hours,
            on_demand_hourly=on_demand_hourly,
            spot_hourly=spot_hourly,
            on_demand_total=on_demand_total,
            spot_total=spot_total,
            use_spot=use_spot,
        )

    def estimate_from_params(
        self,
        instance_type: str,
        instance_count: int = 1,
        max_runtime_hours: float = 24.0,
        use_spot: bool = True,
    ) -> CostEstimate:
        """Estimate cost from individual parameters.

        Args:
            instance_type: SageMaker instance type.
            instance_count: Number of instances.
            max_runtime_hours: Maximum runtime in hours.
            use_spot: Whether to use spot instances.

        Returns:
            CostEstimate with pricing information.
        """
        hourly_price = self.get_hourly_price(instance_type)

        on_demand_hourly = hourly_price * instance_count
        spot_hourly = on_demand_hourly * SPOT_DISCOUNT_FACTOR

        on_demand_total = on_demand_hourly * max_runtime_hours
        spot_total = spot_hourly * max_runtime_hours

        return CostEstimate(
            instance_type=instance_type,
            instance_count=instance_count,
            max_runtime_hours=max_runtime_hours,
            on_demand_hourly=on_demand_hourly,
            spot_hourly=spot_hourly,
            on_demand_total=on_demand_total,
            spot_total=spot_total,
            use_spot=use_spot,
        )

    def list_instances(self, gpu_only: bool = True) -> list[tuple[str, float]]:
        """List available instance types with prices.

        Args:
            gpu_only: If True, only list GPU instances.

        Returns:
            List of (instance_type, hourly_price) tuples, sorted by price.
        """
        gpu_prefixes = ("ml.g4", "ml.g5", "ml.p3", "ml.p4", "ml.p5")
        instances = []

        for instance_type, price in self._price_cache.items():
            if gpu_only and not any(instance_type.startswith(p) for p in gpu_prefixes):
                continue
            instances.append((instance_type, price))

        return sorted(instances, key=lambda x: x[1])


def format_instance_table(estimator: CostEstimator | None = None) -> str:
    """Format instance pricing as a table.

    Args:
        estimator: Optional estimator to use (creates new one if None).

    Returns:
        Formatted table string.
    """
    estimator = estimator or CostEstimator()
    instances = estimator.list_instances(gpu_only=False)

    lines = [
        "SageMaker Instance Pricing (US East 1)",
        "-" * 50,
        f"{'Instance Type':<25} {'On-Demand':<12} {'~Spot':<12}",
        "-" * 50,
    ]

    for instance_type, price in instances:
        spot_price = price * SPOT_DISCOUNT_FACTOR
        lines.append(f"{instance_type:<25} ${price:<11.3f} ${spot_price:<11.3f}")

    lines.append("-" * 50)
    lines.append("Note: Spot prices are estimates (~65% discount)")
    return "\n".join(lines)

