"""Tests for cost estimation."""

import pytest

from soen_toolkit.cloud.cost import CostEstimate, CostEstimator, format_instance_table


class TestCostEstimator:
    """Test cost estimation functionality."""

    def test_get_known_instance_price(self):
        """Test getting price for known instance type."""
        estimator = CostEstimator()
        price = estimator.get_hourly_price("ml.g5.xlarge")
        assert price == 1.006

    def test_get_unknown_instance_raises(self):
        """Test that unknown instance type raises error."""
        estimator = CostEstimator()
        with pytest.raises(ValueError) as exc_info:
            estimator.get_hourly_price("ml.unknown.instance")
        assert "Unknown instance type" in str(exc_info.value)

    def test_estimate_from_params_single_instance(self):
        """Test cost estimate for single instance."""
        estimator = CostEstimator()
        estimate = estimator.estimate_from_params(
            instance_type="ml.g5.xlarge",
            instance_count=1,
            max_runtime_hours=2.0,
            use_spot=True,
        )

        assert estimate.instance_type == "ml.g5.xlarge"
        assert estimate.instance_count == 1
        assert estimate.max_runtime_hours == 2.0
        assert estimate.use_spot is True

        # Check pricing calculations
        # ml.g5.xlarge = $1.006/hr
        assert estimate.on_demand_hourly == pytest.approx(1.006)
        assert estimate.on_demand_total == pytest.approx(2.012)  # 1.006 * 2 hours

        # Spot is ~35% of on-demand
        assert estimate.spot_hourly < estimate.on_demand_hourly
        assert estimate.spot_total < estimate.on_demand_total

    def test_estimate_from_params_multi_instance(self):
        """Test cost estimate for multiple instances."""
        estimator = CostEstimator()
        estimate = estimator.estimate_from_params(
            instance_type="ml.g5.xlarge",
            instance_count=4,
            max_runtime_hours=1.0,
            use_spot=False,
        )

        # 4 instances at $1.006/hr each
        assert estimate.on_demand_hourly == pytest.approx(4.024)
        assert estimate.on_demand_total == pytest.approx(4.024)  # 1 hour

    def test_estimated_cost_uses_spot_when_enabled(self):
        """Test that estimated_cost property uses correct pricing."""
        estimator = CostEstimator()

        spot_estimate = estimator.estimate_from_params(
            instance_type="ml.g5.xlarge",
            instance_count=1,
            max_runtime_hours=1.0,
            use_spot=True,
        )
        assert spot_estimate.estimated_cost == spot_estimate.spot_total

        on_demand_estimate = estimator.estimate_from_params(
            instance_type="ml.g5.xlarge",
            instance_count=1,
            max_runtime_hours=1.0,
            use_spot=False,
        )
        assert on_demand_estimate.estimated_cost == on_demand_estimate.on_demand_total

    def test_list_gpu_instances(self):
        """Test listing GPU instances."""
        estimator = CostEstimator()
        instances = estimator.list_instances(gpu_only=True)

        # Should only include GPU instances
        for instance_type, _ in instances:
            assert any(
                instance_type.startswith(p)
                for p in ("ml.g4", "ml.g5", "ml.p3", "ml.p4", "ml.p5")
            )

        # Should be sorted by price
        prices = [price for _, price in instances]
        assert prices == sorted(prices)

    def test_list_all_instances(self):
        """Test listing all instances including CPU."""
        estimator = CostEstimator()
        gpu_instances = estimator.list_instances(gpu_only=True)
        all_instances = estimator.list_instances(gpu_only=False)

        # All instances should include CPU instances
        assert len(all_instances) > len(gpu_instances)

        # Should include CPU instance types
        cpu_types = [t for t, _ in all_instances if t.startswith(("ml.m5", "ml.c5"))]
        assert len(cpu_types) > 0


class TestCostEstimate:
    """Test CostEstimate dataclass."""

    def test_format_output(self):
        """Test that format() produces readable output."""
        estimate = CostEstimate(
            instance_type="ml.g5.xlarge",
            instance_count=1,
            max_runtime_hours=24.0,
            on_demand_hourly=1.006,
            spot_hourly=0.352,
            on_demand_total=24.144,
            spot_total=8.45,
            use_spot=True,
        )

        formatted = estimate.format()

        assert "Cost Estimate:" in formatted
        assert "ml.g5.xlarge" in formatted
        assert "24.0 hours" in formatted
        assert "$1.006" in formatted
        assert "Spot" in formatted

    def test_succeeded_property(self):
        """Test that succeeded checks status correctly."""
        from soen_toolkit.cloud.jobs.base import JobResult, JobStatus

        success = JobResult(job_name="test", status=JobStatus.COMPLETED)
        assert success.succeeded is True

        failed = JobResult(job_name="test", status=JobStatus.FAILED)
        assert failed.succeeded is False


class TestFormatInstanceTable:
    """Test instance table formatting."""

    def test_format_table(self):
        """Test that table is properly formatted."""
        table = format_instance_table()

        # Should have header
        assert "Instance Type" in table
        assert "On-Demand" in table

        # Should have instance rows
        assert "ml.g5.xlarge" in table

        # Should have footer note
        assert "Spot prices are estimates" in table

