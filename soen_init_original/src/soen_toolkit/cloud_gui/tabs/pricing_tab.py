"""Pricing information tab."""

from __future__ import annotations

import logging

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class PricingTab(QWidget):
    """Tab for viewing instance pricing."""

    def __init__(self) -> None:
        super().__init__()
        self._setup_ui()
        self._load_pricing()

    def _setup_ui(self) -> None:
        """Set up the UI."""
        layout = QVBoxLayout(self)

        # Header
        layout.addWidget(QLabel("SageMaker Instance Pricing (US East 1)"))
        layout.addWidget(QLabel("Prices are estimates and may vary. Spot prices fluctuate."))

        # Pricing table
        self.pricing_table = QTableWidget()
        self.pricing_table.setColumnCount(5)
        self.pricing_table.setHorizontalHeaderLabels([
            "Instance Type",
            "GPUs",
            "On-Demand ($/hr)",
            "Spot (~$/hr)",
            "8hr Spot Cost",
        ])
        self.pricing_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.pricing_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.pricing_table.setAlternatingRowColors(True)

        # Make columns resize nicely
        header = self.pricing_table.horizontalHeader()
        if header is not None:
            header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
            for i in range(1, 5):
                header.setSectionResizeMode(i, QHeaderView.ResizeMode.ResizeToContents)

        layout.addWidget(self.pricing_table)

        # Notes
        notes = QLabel(
            "Notes:\n"
            "• Spot instances are typically 50-70% cheaper than on-demand\n"
            "• Spot instances can be interrupted if AWS needs capacity\n"
            "• G5 instances use NVIDIA A10G GPUs (good for most training)\n"
            "• P4d instances use NVIDIA A100 GPUs (high-performance)\n"
            "• P5 instances use NVIDIA H100 GPUs (maximum performance)"
        )
        notes.setWordWrap(True)
        layout.addWidget(notes)

    def _load_pricing(self) -> None:
        """Load pricing data into table."""
        try:
            from soen_toolkit.cloud.cost import SPOT_DISCOUNT_FACTOR, CostEstimator

            estimator = CostEstimator()

            # Define GPU counts for each instance
            gpu_counts = {
                "ml.g4dn.xlarge": 1,
                "ml.g4dn.2xlarge": 1,
                "ml.g4dn.4xlarge": 1,
                "ml.g4dn.8xlarge": 1,
                "ml.g4dn.12xlarge": 4,
                "ml.g4dn.16xlarge": 1,
                "ml.g5.xlarge": 1,
                "ml.g5.2xlarge": 1,
                "ml.g5.4xlarge": 1,
                "ml.g5.8xlarge": 1,
                "ml.g5.12xlarge": 4,
                "ml.g5.16xlarge": 1,
                "ml.g5.24xlarge": 4,
                "ml.g5.48xlarge": 8,
                "ml.p3.2xlarge": 1,
                "ml.p3.8xlarge": 4,
                "ml.p3.16xlarge": 8,
                "ml.p4d.24xlarge": 8,
                "ml.p4de.24xlarge": 8,
                "ml.p5.48xlarge": 8,
                "ml.m5.large": 0,
                "ml.m5.xlarge": 0,
                "ml.m5.2xlarge": 0,
                "ml.m5.4xlarge": 0,
                "ml.c5.large": 0,
                "ml.c5.xlarge": 0,
                "ml.c5.2xlarge": 0,
            }

            instances = estimator.list_instances(gpu_only=False)

            self.pricing_table.setRowCount(len(instances))

            for row, (instance_type, on_demand) in enumerate(instances):
                spot = on_demand * SPOT_DISCOUNT_FACTOR
                spot_8hr = spot * 8

                gpus = gpu_counts.get(instance_type, "?")

                self.pricing_table.setItem(row, 0, QTableWidgetItem(instance_type))
                self.pricing_table.setItem(row, 1, QTableWidgetItem(str(gpus)))

                on_demand_item = QTableWidgetItem(f"${on_demand:.3f}")
                on_demand_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.pricing_table.setItem(row, 2, on_demand_item)

                spot_item = QTableWidgetItem(f"${spot:.3f}")
                spot_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.pricing_table.setItem(row, 3, spot_item)

                cost_8hr_item = QTableWidgetItem(f"${spot_8hr:.2f}")
                cost_8hr_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                self.pricing_table.setItem(row, 4, cost_8hr_item)

        except Exception:
            logger.exception("Failed to load pricing")

