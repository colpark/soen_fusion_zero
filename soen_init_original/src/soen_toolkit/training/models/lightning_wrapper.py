"""Lightning wrapper for SOEN models.

This module provides a PyTorch Lightning wrapper for SOEN models to enable
training with the Lightning ecosystem.
"""

import inspect  # Added for dynamic loss argument handling
import logging
from pathlib import Path
import tempfile
from typing import Any
from uuid import uuid4

import pytorch_lightning as pl
import torch
from torch import nn

# Import SOEN-specific modules
from soen_toolkit.core import SOENModelCore
from soen_toolkit.core.model_yaml import build_model_from_yaml
from soen_toolkit.training.callbacks.metrics import METRICS_REGISTRY

# Experiment configuration dataclass (for type hints)
from soen_toolkit.training.configs.config_classes import (
    ExperimentConfig,
)

# Classifier wrapper removed; LightningModule handles time pooling + scaling.
from soen_toolkit.training.losses import (
    LOSS_REGISTRY,
)  # Registry of custom loss functions

logger = logging.getLogger(__name__)


class SOENLightningModule(pl.LightningModule):
    """PyTorch Lightning wrapper for SOEN models.

    This class wraps a SOEN core model and provides the necessary
    methods for training with PyTorch Lightning. It also implements
    time pooling and optional scaling on the SOEN outputs.

    Attributes:
        config: Configuration object for the model and training
        model: SOENModelCore instance (the core model)
        loss_fn: Loss function for training

    """

    def __init__(self, config: ExperimentConfig) -> None:
        """Initialize SOENLightningModule.

        Args:
            config: Configuration object containing model and training settings

        """
        super().__init__()
        self.config: ExperimentConfig = config
        self.model: SOENModelCore = self._build_model()
        self._created_local_pg: bool = False

        # g tracking will be enabled conditionally if a loss requires it

        # Parse and store time pooling configuration on the LightningModule itself
        tp_cfg = self.config.model.time_pooling
        if isinstance(tp_cfg, dict):
            self.time_pooling_method_name = tp_cfg.get("name", "max")
            self.time_pooling_params = tp_cfg.get("params", {})
        else:
            self.time_pooling_method_name = tp_cfg
            self.time_pooling_params = {}
        if "scale" not in self.time_pooling_params:
            self.time_pooling_params["scale"] = 1.0
        self.range_start = getattr(self.config.model, "range_start", None)
        self.range_end = getattr(self.config.model, "range_end", None)
        self.autoregressive = getattr(self.config.training, "autoregressive", False)

        # Latest processed state cache (for callbacks/aux losses)
        self.latest_processed_state: torch.Tensor | None = None
        self.latest_final_state: torch.Tensor | None = None
        self.latest_all_states: list[torch.Tensor] | None = None
        self.latest_inputs: torch.Tensor | None = None  # For losses that need to rerun forward (e.g., local_expansion_loss)

        # Pending initial states for stateful training (set by callback)
        self._pending_initial_states: dict[int, torch.Tensor] | None = None
        self._pending_s1_states: dict[int, torch.Tensor] | None = None
        self._pending_s2_states: dict[int, torch.Tensor] | None = None

        # Initialize (possibly multiple) loss functions
        self._initialize_loss_functions()

        # These metric dictionaries are placeholders for future extensions
        self.train_metrics: dict[str, Any] = {}
        self.val_metrics: dict[str, Any] = {}
        self.test_metrics: dict[str, Any] = {}
        self._validation_step_outputs: list[Any] = []
        self._test_step_outputs: list[Any] = []
        self._training_step_outputs: list[Any] = []  # Store training outputs for epoch-end metrics

        # TBPTT: expose chunk length on the module for PL versions that read it
        self.truncated_bptt_steps: int | None = None
        if getattr(self.config.training, "use_tbptt", False):
            tbptt_steps = self.config.training.tbptt_steps
            if tbptt_steps is not None:
                self.truncated_bptt_steps = int(tbptt_steps)

        # Save hyperparameters for checkpoint resumption
        # This is needed for train_from_checkpoint functionality to work correctly
        if hasattr(self.config, "to_dict"):
            try:
                self.save_hyperparameters(self.config.to_dict(serializable=True))
                logger.debug("Saved hyperparameters to checkpoint for resumption support")
            except TypeError:
                try:
                    self.save_hyperparameters(self.config.to_dict())
                    logger.debug("Saved hyperparameters to checkpoint (non-serializable mode)")
                except Exception as e:
                    logger.warning(f"Failed to save hyperparameters: {e}. Checkpoint resumption may not work correctly.")

    def _build_model(self) -> SOENModelCore:
        """Build SOEN model core based on configuration.
        Handles loading exact state vs. rebuilding from config.

        Returns:
            SOENModelCore: The SOEN core model

        """
        soen_model = None  # Initialize

        if self.config.model.base_model_path is not None:
            model_path = Path(self.config.model.base_model_path)
            logger.info(f"Base model path provided: {model_path}")

            # Ensure the file exists
            if not model_path.exists():
                msg = f"Base model file not found: {model_path}"
                raise FileNotFoundError(msg)

            suffix = model_path.suffix.lower()
            # YAML: treat as architecture spec (friendlier UX)
            if suffix in {".yaml", ".yml"}:
                logger.warning(
                    "base_model_path points to a YAML spec; building from YAML. Consider using model.architecture_yaml instead.",
                )
                soen_model = build_model_from_yaml(
                    str(model_path),
                    honor_yaml_seed=False,
                )

            # JSON: try flexible builder first (handles exported JSON and spec-style JSON)
            elif suffix == ".json":
                try:
                    soen_model = SOENModelCore.build(str(model_path))
                    logger.info("Model constructed from JSON (exported or spec-style)")
                except Exception:
                    # Fallback to strict loader for exported JSON format
                    soen_model = SOENModelCore.load(str(model_path))

            # Checkpoint is not valid here
            elif suffix == ".ckpt":
                if self.config.model.load_exact_model_state:
                    msg = "load_exact_model_state=True with a .ckpt is invalid; resume via training.training.train_from_checkpoint."
                    raise ValueError(
                        msg,
                    )
                msg = "base_model_path points to a .ckpt. Use training.training.train_from_checkpoint to resume; do not set base_model_path to .ckpt."
                raise ValueError(
                    msg,
                )

            # Binary saves: .soen/.pth -> load then optionally rebuild
            # Check if loading exact state is requested and path is not a checkpoint
            elif self.config.model.load_exact_model_state:
                logger.info(f"Loading exact SOEN model state from: {model_path}")
                try:
                    soen_model = SOENModelCore.load(str(model_path))
                    logger.info(
                        f"Exact SOEN model loaded successfully: {len(soen_model.layers)} layers",
                    )
                except Exception as e:
                    # Fail-fast: loading exact model state must be deterministic. Silent fallback to a
                    # different file can mask corruption or path mistakes.
                    raise RuntimeError(
                        f"Failed to load exact SOEN model state from {model_path}. "
                        "If this file is corrupted/truncated, re-export or re-train and point "
                        "base_model_path to the correct artifact."
                    ) from e
            else:
                # Default: Load config from model file and rebuild
                logger.info(
                    f"Loading SOEN model configuration from: {model_path} and rebuilding.",
                )
                try:
                    base_model_for_config = SOENModelCore.load(str(model_path))
                    logger.info(f"Configuration loaded successfully from: {model_path}")
                except Exception as e:
                    logger.exception(
                        f"Failed to load base model for configuration extraction: {e}",
                    )
                    raise

                # Extract configurations
                sim_config = base_model_for_config.sim_config
                layers_config = base_model_for_config.layers_config
                connections_config = base_model_for_config.connections_config

                # Apply model modification function if provided (Optional)
                if hasattr(self.config, "model_modifier_fn") and self.config.model_modifier_fn is not None:
                    logger.info("Applying model modification function")
                    sim_config, layers_config, connections_config = self.config.model_modifier_fn(
                        sim_config,
                        layers_config,
                        connections_config,
                    )

                # Create new SOEN model with potentially modified configs
                soen_model = SOENModelCore(
                    sim_config=sim_config,
                    layers_config=layers_config,
                    connections_config=connections_config,
                )
                logger.info("Rebuilt SOEN model from extracted configuration.")

        else:
            # Build from YAML architecture if provided
            arch_yaml = getattr(self.config.model, "architecture_yaml", None)
            arch_inline = getattr(self.config.model, "architecture", None)
            if arch_yaml is not None:
                logger.info(f"Building SOEN model from YAML: {arch_yaml}")
                # Training seed should supersede any YAML-level seed
                soen_model = build_model_from_yaml(arch_yaml, honor_yaml_seed=False)
            elif arch_inline is not None:
                logger.info("Building SOEN model from inline architecture dict")
                soen_model = build_model_from_yaml(arch_inline, honor_yaml_seed=False)
            else:
                msg = "No base_model_path or architecture_yaml/architecture provided in configuration."
                raise ValueError(
                    msg,
                )

        # Ensure soen_model is valid before proceeding
        if soen_model is None:
            msg = "SOEN model core (soen_model) was not successfully loaded or built."
            raise RuntimeError(
                msg,
            )

        # Validate model output dimension matches dataset num_classes (for classification tasks)
        self._validate_model_output_dimension(soen_model)

        # Optionally override the simulation time-step dt
        if self.config.model.dt is not None:
            try:
                soen_model.set_dt(self.config.model.dt, propagate_to_layers=True)
                logger.info(
                    f"Overrode SOEN model dt to {self.config.model.dt} (dimensionless units) via config",
                )
            except Exception as e:
                logger.exception(f"Failed to override dt on SOEN model: {e}")

        # Set learnability of dt if specified
        try:
            soen_model.set_dt_learnable(
                self.config.model.dt_learnable,
                propagate_to_layers=True,
            )
        except Exception as e:
            logger.exception(f"Failed to set dt learnability: {e}")

        return soen_model

    def _log_gravity_quantization_codebook(self, params: dict) -> None:
        """Log the codebook used for gravity quantization loss to the training log file.
        This is called once at the beginning of training when gravity quantization loss is detected.
        """
        import logging

        from soen_toolkit.training.losses.loss_functions import (
            _generate_uniform_codebook,
        )

        logger = logging.getLogger(__name__)

        # Generate codebook using the same logic as the loss function
        codebook = params.get("codebook")

        if codebook is None:
            min_val = params.get("min_val")
            max_val = params.get("max_val")
            # Accept num_levels, levels, or bits (bits -> 2**bits)
            num_levels = params.get("num_levels")
            levels = params.get("levels")
            bits = params.get("bits")
            eff_levels = None
            if num_levels is not None:
                eff_levels = int(num_levels)
            elif levels is not None:
                eff_levels = int(levels)
            elif bits is not None:
                b = int(bits)
                if b < 0:
                    msg = "bits must be non-negative"
                    raise ValueError(msg)
                # Bits EXCLUDE zero; our generator always includes a single zero.
                eff_levels = (2**b) + 1

            if min_val is None or max_val is None or eff_levels is None:
                codebook = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0]  # Default
                logger.info("GRAVITY_QUANTIZATION_CODEBOOK: Using default codebook (no parameters specified)")
            else:
                # Use the robustness tool's algorithm (the only method now)
                codebook = _generate_uniform_codebook(min_val, max_val, eff_levels)
                logger.info(f"GRAVITY_QUANTIZATION_CODEBOOK: Generated codebook with min_val={min_val}, max_val={max_val}, levels={eff_levels}")
        else:
            logger.info("GRAVITY_QUANTIZATION_CODEBOOK: Using explicit codebook from configuration")

        # Log the actual codebook values
        codebook_str = ", ".join([f"{val:.6f}" for val in codebook])
        logger.info(f"GRAVITY_QUANTIZATION_CODEBOOK: Codebook values = [{codebook_str}]")
        logger.info(f"GRAVITY_QUANTIZATION_CODEBOOK: Number of levels = {len(codebook)}")

        # Also log other relevant parameters
        factor = params.get("factor", 1.0)
        logger.info(f"GRAVITY_QUANTIZATION_CODEBOOK: Loss scaling factor = {factor}")
        mode = params.get("mode", "mae")
        logger.info(f"GRAVITY_QUANTIZATION_CODEBOOK: Error mode = {mode}")

    def _initialize_loss_functions(self) -> None:
        """Initialise loss functions based on the training configuration."""
        training_config = self.config.training
        self.active_loss_components = []

        # Unified list style only
        if not training_config.loss or not training_config.loss.losses:
            logger.warning("No losses configured. Training will have zero total loss unless configured.")
            return

        for item in training_config.loss.losses:
            name = item.name.lower()
            if name not in LOSS_REGISTRY:
                logger.warning(f"Loss '{name}' not found in registry; skipping.")
                continue

            # Conditionally enable g tracking if needed by configured losses
            if name == "ensure_post_sample_decay":
                try:
                    self.model.set_tracking(track_g=True)
                except Exception as e:
                    logger.warning(f"Failed to enable g tracking for '{name}': {e}")

            raw_fn = LOSS_REGISTRY[name]
            fn_inst = raw_fn(**item.params) if isinstance(raw_fn, type) and issubclass(raw_fn, nn.Module) else raw_fn

            self.active_loss_components.append(
                {
                    "name": name,
                    "fn": fn_inst,
                    "weight": item.weight,
                    "params_from_config": item.params,
                },
            )

            # Log gravity quantization codebook if this loss is being used
            if name == "gravity_quantization_loss":
                self._log_gravity_quantization_codebook(item.params)

    def _compute_total_loss(
        self,
        y_hat_logits: torch.Tensor,
        y_targets: torch.Tensor,
        processed_output_states: torch.Tensor | None,
        stage_prefix: str,
    ) -> torch.Tensor:
        """Compute total weighted loss (base + auxiliary) and log individual components."""
        # IMPORTANT: Don't in-place accumulate into a leaf tensor without grad.
        # Start with None and build the graph using out-of-place additions.
        total_loss: torch.Tensor | None = None

        # Determine which loss components to use for this step
        comps = self.active_loss_components
        if stage_prefix == "train" and self.config.training.alternate_losses and len(comps) > 1:
            batches_per_loss = self.config.training.batches_per_loss
            idx = (self.global_step // batches_per_loss) % len(comps)
            comps = [comps[idx]]

        for comp in comps:
            loss_fn = comp["fn"]
            weight = comp["weight"]
            cfg_params = comp["params_from_config"]
            loss_name = comp["name"]

            # For nn.Module instances, we need to inspect the forward method signature
            # instead of the __call__ method signature
            if isinstance(loss_fn, nn.Module):
                sig = inspect.signature(loss_fn.forward)
            else:
                sig = inspect.signature(loss_fn)  # type: ignore[arg-type]
            call_kwargs: dict[str, Any] = {}

            # Map standard arg names if present in function signature
            if "outputs" in sig.parameters:
                call_kwargs["outputs"] = y_hat_logits
            if "logits" in sig.parameters:  # For pooled_rich_margin_loss
                call_kwargs["logits"] = processed_output_states
            if "targets" in sig.parameters:
                call_kwargs["targets"] = y_targets
            if "model" in sig.parameters:
                call_kwargs["model"] = self
            if "processed_output_states" in sig.parameters:
                if processed_output_states is None:
                    logger.warning(
                        f"Loss '{loss_name}' requires 'processed_output_states', but it is None. Skipping this component.",
                    )
                    continue
                call_kwargs["processed_output_states"] = processed_output_states

            # Handle autoregressive-specific parameters
            if loss_name in ["autoregressive_loss", "autoregressive_cross_entropy"]:
                if self.config.training.autoregressive:
                    # Pass sequence outputs for autoregressive loss
                    if "sequence_outputs" in sig.parameters:
                        call_kwargs["sequence_outputs"] = y_hat_logits

                    # Determine effective time_steps_per_token for loss
                    # If we pooled in forward(), the outputs are already at token level (1 step/token)
                    # If we didn't pool (time_steps_per_token=1), it's also 1 step/token
                    # So we always pass 1 to the loss function if we are in the standard flow
                    # However, if we want to support the old way where loss handles striding,
                    # we need to check if we pooled.

                    ar_cfg = self.config.training.ar
                    if ar_cfg is None:
                        pass
                    else:
                        ar_cfg.get("time_steps_per_token", 1) if isinstance(ar_cfg, dict) else ar_cfg.time_steps_per_token

                    # If tspt > 1, we pooled in forward(), so effective steps for loss is 1
                    # If tspt == 1, effective steps is 1
                    # So we always pass 1 here because forward() handles the complexity
                    if "time_steps_per_token" in sig.parameters:
                        call_kwargs["time_steps_per_token"] = 1

                    if "start_timestep" in sig.parameters:
                        call_kwargs["start_timestep"] = self.config.training.autoregressive_start_timestep
                    if "vocab_size" in sig.parameters:
                        call_kwargs["vocab_size"] = self.config.data.num_classes
                else:
                    # Skip autoregressive loss if not in autoregressive mode
                    logger.warning(f"Skipping {loss_name} because autoregressive=False")
                    continue

            # Add any loss-specific parameters defined in the config
            # But only for non-nn.Module functions (classes are already instantiated with params)
            if not isinstance(loss_fn, nn.Module):
                for p_name, p_val in cfg_params.items():  # type: ignore[union-attr, operator]
                    if p_name in sig.parameters and p_name not in call_kwargs:
                        call_kwargs[p_name] = p_val
                    elif p_name not in sig.parameters and not isinstance(
                        loss_fn,
                        (
                            nn.CrossEntropyLoss,
                            nn.MSELoss,
                            nn.BCEWithLogitsLoss,
                            nn.L1Loss,
                            nn.HuberLoss,
                        ),
                    ):
                        logger.warning(
                            f"Parameter '{p_name}' for custom loss '{loss_name}' not present in signature {sig}. Ignoring.",
                        )

            # ----------------------------------------------------------------
            # Compute the loss value
            # ----------------------------------------------------------------
            try:
                # --- START FIX ---
                # Priority 1: If in autoregressive mode and the user specifies 'cross_entropy',
                # automatically route to the correct autoregressive implementation.
                if self.config.training.autoregressive and loss_name == "cross_entropy":
                    from soen_toolkit.training.losses import (
                        autoregressive_cross_entropy,
                    )

                    # The `y_targets` passed to this function is already the correctly
                    # shifted `y_targets_for_loss` from the training/validation step.
                    current_loss_val = autoregressive_cross_entropy(
                        y_hat_logits,
                        y_targets,
                    )

                # Priority 2: Handle standard PyTorch nn.Module losses (legacy support)
                elif isinstance(
                    loss_fn,
                    (
                        nn.CrossEntropyLoss,
                        nn.MSELoss,
                        nn.BCEWithLogitsLoss,
                        nn.L1Loss,
                        nn.HuberLoss,
                    ),
                ):
                    # This autoregressive check here is now mostly for backwards compatibility
                    if isinstance(loss_fn, nn.CrossEntropyLoss) and self.config.training.autoregressive:
                        from soen_toolkit.training.losses import (
                            autoregressive_cross_entropy,
                        )

                        current_loss_val = autoregressive_cross_entropy(
                            y_hat_logits,
                            y_targets,
                        )
                    else:
                        current_loss_val = loss_fn(y_hat_logits, y_targets)

                # Priority 3: Handle all other registered functional losses
                else:
                    current_loss_val = loss_fn(**call_kwargs)  # type: ignore[operator, assignment]
                # --- END FIX ---

            except TypeError as e:
                logger.exception(
                    f"TypeError when calling loss function '{loss_name}': {e}. Signature: {sig}. KWArgs passed: {list(call_kwargs.keys())}",
                )
                raise

            weighted_loss = weight * current_loss_val  # type: ignore[operator]

            # Log raw and weighted loss components. For training we log both per
            # step and per epoch so that epoch-averaged curves can be shown
            self.log(
                f"{stage_prefix}_loss/{loss_name}",
                current_loss_val.detach(),
                prog_bar=False,
                sync_dist=True,
                on_step=(stage_prefix == "train"),  # CORRECT: Log step metrics ONLY for training
                on_epoch=True,  # CORRECT: Log epoch-averaged metrics for ALL stages
            )
            if weight != 1.0:
                self.log(
                    f"{stage_prefix}_loss/{loss_name}_w",
                    weighted_loss.detach(),  # type: ignore[union-attr]
                    prog_bar=False,
                    sync_dist=True,
                    on_step=(stage_prefix == "train"),  # CORRECT: Log step metrics ONLY for training
                    on_epoch=True,  # CORRECT: Log epoch-averaged metrics for ALL stages
                )

            total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss  # type: ignore[assignment, operator]

        # Log total loss
        # When logging training metrics the viewer expects an epoch aggregated
        # series to be available. Setting ``on_epoch=True`` ensures PyTorch
        # Lightning emits an additional ``*_epoch`` scalar at the end of each
        # epoch, which the TensorBoard viewer can use for the epoch X‑axis.
        # If no loss components were active, return a zero that still allows backward
        if total_loss is None:
            total_loss = torch.zeros((), device=self.device, requires_grad=True)

        self.log(
            f"{stage_prefix}_loss/total",
            total_loss,
            prog_bar=True,
            sync_dist=True,
            on_step=(stage_prefix == "train"),  # CORRECT: Log step metrics ONLY for training
            on_epoch=True,  # CORRECT: Log epoch-averaged metrics for ALL stages
        )
        return total_loss

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor]]:
        """Forward pass through the model.

        Args:
            x: Input tensor [batch_size, seq_len, input_dim]

        Returns:
            Tuple containing:
                - output: Pooled features or full sequence features (if autoregressive)
                - final_state: State history of final layer [batch_size, seq_len+1, dim]
                - all_states: List of state histories for all layers

        """
        # Cache inputs for losses that need them (e.g., local_expansion_loss)
        self.latest_inputs = x

        # Check if callback has set initial states for stateful training
        initial_states = getattr(self, '_pending_initial_states', None)
        s1_states = getattr(self, '_pending_s1_states', None)
        s2_states = getattr(self, '_pending_s2_states', None)

        if initial_states is not None or s1_states is not None or s2_states is not None:
            final_state, all_states = self.model(
                x,
                initial_states=initial_states,
                s1_inits=s1_states,
                s2_inits=s2_states
            )
            # Clear after use to avoid reusing states incorrectly
            self._pending_initial_states = None
            self._pending_s1_states = None
            self._pending_s2_states = None
        else:
            final_state, all_states = self.model(x)

        # Expose raw histories for custom/self-supervised losses
        self.latest_final_state = final_state
        self.latest_all_states = all_states

        # Autoregressive pooling (if enabled)
        if self.config.training.autoregressive:
            # Get AR config (handle both new and old formats)
            ar_cfg = self.config.training.ar
            if ar_cfg is None:
                # Fallback to old format
                time_steps_per_token = self.config.training.time_steps_per_token
                token_pooling = {"method": "final", "params": {}}
            # New format
            elif isinstance(ar_cfg, dict):
                # Should be converted by dataclass, but handle dict just in case
                time_steps_per_token = ar_cfg.get("time_steps_per_token", 1)
                token_pooling = ar_cfg.get("token_pooling", {"method": "final"})
            else:
                time_steps_per_token = ar_cfg.time_steps_per_token
                token_pooling = ar_cfg.token_pooling

            if time_steps_per_token > 1:
                from soen_toolkit.training.utils.ar_helpers import pool_token_timesteps

                method: str = token_pooling.get("method", "final")  # type: ignore[assignment]
                params: dict[str, Any] | None = token_pooling.get("params", {})  # type: ignore[assignment]

                # Pool timesteps within each token -> [batch, num_tokens, output_dim]
                # Note: final_state includes initial state at t=0, so we need to be careful
                # The pooling function expects [batch, total_timesteps, dim]
                # We skip t=0 for pooling
                seq_outputs = final_state[:, 1:, :]
                pooled_output = pool_token_timesteps(
                    seq_outputs,
                    time_steps_per_token,
                    method,
                    params
                )

                # For consistency with other paths, we might want to prepend a dummy initial state
                # or just return the pooled sequence.
                # Since this is specifically for AR prediction, we return the pooled sequence.
                processed_value = pooled_output
            else:
                # Standard AR (1 timestep/token) - return full sequence (skipping t=0)
                processed_value = final_state[:, 1:, :]
        # Standard time pooling for classification/regression
        # For "none" pooling (distillation), include t=0 to match teacher trajectory
        elif self.time_pooling_method_name == "none":
            processed_value = self.process_output(final_state)
        else:
            processed_value = self.process_output(final_state[:, 1:, :])

        self.latest_processed_state = processed_value
        return processed_value, final_state, all_states

    def process_output(self, state_history: torch.Tensor) -> torch.Tensor:
        """Process state history using the specified method.

        Args:
            state_history: Tensor of shape [batch_size, timesteps, state_dim] from the last layer

        Returns:
            Processed tensor of shape [batch_size, state_dim] for pooling methods,
            or [batch_size, timesteps, state_dim] when method is "none".

        """
        _batch_size, total_timesteps, _state_dim = state_history.shape

        method_name = self.time_pooling_method_name
        method_scale = float(self.time_pooling_params.get("scale", 1.0))

        # ------------------------------------------------------------------
        # Standard processing methods
        # ------------------------------------------------------------------
        if method_name == "max":
            processed_value = torch.max(state_history, dim=1)[0]

        elif method_name == "mean":
            processed_value = torch.mean(state_history, dim=1)

        elif method_name == "rms":
            processed_value = torch.sqrt(torch.mean(state_history**2, dim=1) + 1e-8)

        elif method_name == "mean_range":
            # Default to last 50 timesteps if neither range_start nor range_end specified
            default_points = min(50, total_timesteps)

            start_idx = self.range_start if self.range_start is not None else max(0, total_timesteps - default_points)
            end_idx = self.range_end if self.range_end is not None else total_timesteps

            start_idx = max(0, min(start_idx, total_timesteps - 1))
            end_idx = max(start_idx + 1, min(end_idx, total_timesteps))

            if start_idx >= end_idx:
                logger.warning(
                    f"Invalid range [{start_idx}:{end_idx}] for mean_range. Using full sequence mean.",
                )
                processed_value = torch.mean(state_history, dim=1)
            else:
                processed_value = torch.mean(state_history[:, start_idx:end_idx, :], dim=1)

        elif method_name == "final":
            # Use final timestep
            processed_value = state_history[:, -1, :]

        elif method_name == "mean_last_n":
            n = int(self.time_pooling_params.get("n", 1))
            n = max(1, min(n, total_timesteps))
            processed_value = torch.mean(state_history[:, -n:, :], dim=1)

        elif method_name == "ewa":
            # Exponentially Weighted Average over timesteps
            # Define weights that decay from 1.0 at the latest timestep to min_weight at the earliest
            min_weight = float(self.time_pooling_params.get("min_weight", 0.2))
            min_weight = max(1e-6, min(min_weight, 1.0))
            # Create weights linearly spaced in log-space for stability
            t = torch.linspace(0.0, 1.0, steps=total_timesteps, device=state_history.device)
            weights = min_weight ** (1.0 - t)
            weights = weights / weights.sum()
            # Apply weights across time dimension
            processed_value = torch.einsum("btf,t->bf", state_history, weights)

        elif method_name == "none":
            # No pooling - return full state history for seq2seq tasks (e.g., distillation)
            # Output shape: [batch_size, timesteps, state_dim]
            # Scaling is applied to 3D tensor as well
            processed_value = state_history

        else:
            msg = f"Unknown time pooling method: {method_name}. Supported: max, mean, rms, mean_range, final, mean_last_n, ewa, none"
            raise ValueError(
                msg,
            )

        # Apply optional scaling (works for both 2D and 3D tensors)
        if method_scale != 1.0:
            processed_value = processed_value * method_scale

        return processed_value

    def _pool_distillation_sequences(
        self,
        student_seq: torch.Tensor,
        teacher_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Optionally pool student/teacher sequences for distillation."""
        if self.time_pooling_method_name == "none":
            return student_seq, teacher_seq

        if student_seq.dim() != 3 or teacher_seq.dim() != 3:
            msg = "Distillation time pooling expects 3D [B, T, D] sequences for student and teacher."
            raise ValueError(msg)

        pooled_student = self.process_output(student_seq)
        pooled_teacher = self.process_output(teacher_seq)
        # Keep latest_processed_state aligned with what feeds the loss
        self.latest_processed_state = pooled_student
        return pooled_student, pooled_teacher

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Perform a training step.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch

        Returns:
            Dictionary containing loss and other metrics

        """
        x, y_targets = batch

        # Log effective input sequence length seen by the model (epoch-aggregated)
        self.log(
            "input_seq_len",
            float(x.size(1)) if x.dim() >= 2 else float(0),
            prog_bar=False,
            logger=True,
            on_step=False,
            on_epoch=True,
        )

        # Self-supervised reconstruction support: if paradigm == 'self_supervised' or 'unsupervised'
        # and labels are absent or dummy scalars, treat target as input (reconstruction).
        try:
            paradigm = getattr(self.config.training, "paradigm", "supervised")
        except Exception:
            paradigm = "supervised"
        if paradigm in ["unsupervised", "self_supervised"]:
            # Detect lack of real targets: common cases
            # - y_targets is a scalar per-sample (e.g., shape [B] or []) used as placeholder
            # - or y_targets has incompatible shape with outputs and we prefer reconstruction
            # We'll set y_targets = x to drive MSE reconstruction when user configures losses: [ {name: mse} ]
            if not isinstance(y_targets, torch.Tensor) or y_targets.ndim <= 1:
                y_targets = x

        # Manual TBPTT fallback: split inside training_step (supports overlap via tbptt_stride)
        use_tbptt: bool = bool(getattr(self.config.training, "use_tbptt", False))
        split_size: int | None = getattr(self.config.training, "tbptt_steps", None)
        if use_tbptt and isinstance(split_size, int) and split_size > 0 and x.dim() >= 2 and x.size(1) > split_size:
            stride = getattr(self.config.training, "tbptt_stride", None)
            if not isinstance(stride, int) or stride <= 0:
                stride = split_size
            starts = list(range(0, x.size(1), stride))
            if starts and starts[-1] >= x.size(1):
                starts = starts[:-1]
            if not starts:
                starts = [0]
            num_chunks = len(starts)
            accumulated_loss: torch.Tensor | None = None
            last_preds: torch.Tensor | None = None
            last_targets: torch.Tensor | None = None

            for start in starts:
                end = min(start + split_size, x.size(1))
                x_chunk = x[:, start:end, ...]

                # Slice targets if they carry a time dimension; otherwise reuse
                if y_targets.dim() >= 2 and y_targets.size(1) == x.size(1):
                    y_chunk = y_targets[:, start:end, ...]
                else:
                    y_chunk = y_targets

                y_hat_logits, final_state_chunk, _all_states = self(x_chunk)

                # AR targets per chunk or plain
                if self.config.training.autoregressive:
                    from soen_toolkit.training.utils.helpers import (
                        build_autoregressive_targets,
                        extract_token_sequence_from_inputs,
                    )

                    mode = getattr(self.config.training, "autoregressive_mode", "next_token")
                    input_sequence = extract_token_sequence_from_inputs(x_chunk)
                    y_targets_for_loss = build_autoregressive_targets(input_sequence, y_chunk, mode=mode)
                    outputs_for_loss = y_hat_logits
                # Distillation: use full state trajectory including t=0 to match teacher
                elif paradigm == "distillation":
                    outputs_for_loss, y_targets_for_loss = self._pool_distillation_sequences(
                        final_state_chunk,
                        y_chunk,
                    )
                # For self-supervised recon: prefer sequence outputs matching inputs
                elif paradigm in ["unsupervised", "self_supervised"]:
                    y_targets_for_loss = x_chunk
                    # final_state includes initial state at t=0; drop it to match input length
                    outputs_for_loss = final_state_chunk[:, 1:, :]
                else:
                    # If supervised seq2seq and targets are sequence-shaped, use sequence outputs
                    mapping = str(getattr(self.config.training, "mapping", "seq2static")).lower()
                    if mapping == "seq2seq" and isinstance(y_chunk, torch.Tensor) and y_chunk.dim() == 3:
                        y_targets_for_loss = y_chunk
                        outputs_for_loss = final_state_chunk[:, 1:, :]
                    else:
                        y_targets_for_loss = y_chunk
                        outputs_for_loss = y_hat_logits

                processed_states = self.latest_processed_state
                if processed_states is None:
                    processed_states = y_hat_logits

                loss_chunk = self._compute_total_loss(
                    outputs_for_loss,
                    y_targets_for_loss,
                    processed_states,
                    stage_prefix="train",
                )

                accumulated_loss = loss_chunk if accumulated_loss is None else accumulated_loss + loss_chunk
                last_preds = outputs_for_loss.detach()
                last_targets = y_targets_for_loss.detach() if isinstance(y_targets_for_loss, torch.Tensor) else x_chunk.detach()

            assert accumulated_loss is not None
            total_loss = accumulated_loss / float(num_chunks)
            return {
                "loss": total_loss,
                "preds": last_preds if last_preds is not None else x.new_zeros(()),
                "targets": last_targets if last_targets is not None else (y_targets.detach() if isinstance(y_targets, torch.Tensor) else x.detach()),
            }

        # Standard full-sequence path
        y_hat_logits, final_state, all_states = self(x)

        # Unified AR target construction
        if self.config.training.autoregressive:
            from soen_toolkit.training.utils.ar_helpers import build_multistep_ar_targets
            from soen_toolkit.training.utils.helpers import extract_token_sequence_from_inputs

            # Get AR config
            ar_cfg = self.config.training.ar
            if ar_cfg is None:
                mode = getattr(self.config.training, "autoregressive_mode", "next_token")
            else:
                mode = ar_cfg.get("mode", "next_token") if isinstance(ar_cfg, dict) else ar_cfg.mode

            input_sequence = extract_token_sequence_from_inputs(x)

            # Use new unified target builder
            # Note: For multi-timestep, we still want 1 target per token
            y_targets_for_loss = build_multistep_ar_targets(input_sequence)
            outputs_for_loss = y_hat_logits
        # Distillation: use full state trajectory including t=0 to match teacher
        elif paradigm == "distillation":
            outputs_for_loss, y_targets_for_loss = self._pool_distillation_sequences(final_state, y_targets)
        # If self-supervised and target seems placeholder or mismatched, reconstruct inputs
        elif paradigm in ["unsupervised", "self_supervised"]:
            mapping = str(getattr(self.config.training, "mapping", "seq2static")).lower()
            if mapping == "seq2seq":
                # For seq2seq unsupervised, use sequence outputs to reconstruct input sequence
                y_targets_for_loss = x
                outputs_for_loss = final_state[:, 1:, :]
            else:
                # For seq2static unsupervised, use provided targets (should be summary of inputs)
                y_targets_for_loss = y_targets
                outputs_for_loss = y_hat_logits
        else:
            mapping = str(getattr(self.config.training, "mapping", "seq2static")).lower()
            # If supervised seq2seq, use sequence outputs to match sequence targets
            if mapping == "seq2seq" and isinstance(y_targets, torch.Tensor) and (y_targets.dim() == 3 or y_targets.dim() == 2):
                y_targets_for_loss = y_targets
                outputs_for_loss = final_state[:, 1:, :]
            elif isinstance(y_targets, torch.Tensor) and y_targets.dim() == 3:
                # Auto-fix mismatch: if targets are sequences but mapping not set, warn and align automatically
                try:
                    import logging as _logging

                    _logging.getLogger(__name__).warning(
                        "Detected sequence targets (B,T,D) with mapping='%s'. "
                        "Automatically aligning outputs to sequence and dropping initial state. "
                        "Set training.mapping=seq2seq to silence this warning.",
                        mapping,
                    )
                except Exception:
                    pass
                y_targets_for_loss = y_targets
                outputs_for_loss = final_state[:, 1:, :]
            else:
                y_targets_for_loss = y_targets
                outputs_for_loss = y_hat_logits

        # Retrieve processed features (pooled features or sequence features)
        processed_states = self.latest_processed_state
        if processed_states is None:
            processed_states = y_hat_logits

        try:
            total_loss = self._compute_total_loss(
                outputs_for_loss,
                y_targets_for_loss,
                processed_states,
                stage_prefix="train",
            )
        except RuntimeError as e:
            # Provide detailed error message with suggestions
            self._log_detailed_loss_error(e, outputs_for_loss, y_targets_for_loss, "training")
            raise
        # Store outputs for epoch-end training metrics (match JAX backend)
        self._training_step_outputs.append(
            {
                "loss": total_loss,
                "preds": outputs_for_loss.detach(),
                "targets": y_targets.detach() if isinstance(y_targets, torch.Tensor) else x.detach(),
            }
        )
        return {
            "loss": total_loss,
            "preds": outputs_for_loss.detach(),
            "targets": y_targets.detach() if isinstance(y_targets, torch.Tensor) else x.detach(),
        }

    # --- TBPTT support: Let Lightning split long sequences into chunks ---
    def tbptt_split_batch(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        split_size: int,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Split a batch along time dimension for Truncated BPTT.

        Expects inputs shaped [batch, seq_len, ...] and labels aligned accordingly.
        Returns a list of (x_chunk, y_chunk) pairs of length <= ceil(seq_len/split_size).
        """
        x, y = batch
        # Assume time dimension is dim=1 across the project (batch_first=True)
        if x.dim() < 2:
            return [batch]
        T = x.size(1)
        if split_size is None or split_size <= 0 or split_size >= T:
            return [batch]
        stride = getattr(self.config.training, "tbptt_stride", None)
        if not isinstance(stride, int) or stride <= 0:
            stride = split_size
        starts = list(range(0, T, stride))
        if starts and starts[-1] >= T:
            starts = starts[:-1]
        if not starts:
            starts = [0]
        x_chunks = [x[:, s : min(s + split_size, T), ...] for s in starts]
        y_chunks: list[torch.Tensor] = []
        for s in starts:
            e = min(s + split_size, T)
            if y.dim() >= 2 and y.size(1) == T:
                y_chunks.append(y[:, s:e, ...])
            else:
                y_chunks.append(y)
        return list(zip(x_chunks, y_chunks, strict=False))

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        x, y_targets = batch
        y_hat_logits, final_soen_state, _all_soen_states = self(x)

        if self.config.training.autoregressive:
            from soen_toolkit.training.utils.helpers import (
                build_autoregressive_targets,
                extract_token_sequence_from_inputs,
            )

            mode = getattr(self.config.training, "autoregressive_mode", "next_token")
            input_sequence = extract_token_sequence_from_inputs(x)
            y_targets_for_loss = build_autoregressive_targets(input_sequence, y_targets, mode=mode)
        else:
            # Mirror training_step logic: honor mapping and seq2seq targets
            paradigm = getattr(self.config.training, "paradigm", "supervised")
            mapping = str(getattr(self.config.training, "mapping", "seq2static")).lower()
            # If supervised seq2seq and targets carry a time dimension, align to sequence outputs
            if paradigm == "distillation":
                y_hat_logits, y_targets_for_loss = self._pool_distillation_sequences(
                    final_soen_state,
                    y_targets,
                )
            elif paradigm in ["unsupervised", "self_supervised"]:
                if mapping == "seq2seq":
                    # For seq2seq unsupervised, use sequence outputs to reconstruct input sequence
                    y_targets_for_loss = x
                    # final_soen_state includes initial t=0, drop it
                    y_hat_logits = final_soen_state[:, 1:, :]
                else:
                    # For seq2static unsupervised, use provided targets (should be summary of inputs)
                    y_targets_for_loss = y_targets
                    # y_hat_logits already set to pooled outputs
            elif isinstance(y_targets, torch.Tensor) and (y_targets.dim() == 3 or (y_targets.dim() == 2 and mapping == "seq2seq")):
                # Auto-switch to seq2seq when targets are sequences
                y_targets_for_loss = y_targets
                y_hat_logits = final_soen_state[:, 1:, :]
            else:
                y_targets_for_loss = y_targets

        processed_output_states = self.latest_processed_state
        if processed_output_states is None:
            processed_output_states = y_hat_logits

        # Sample0 bar-chart capture is handled by a dedicated callback.

        try:
            total_loss = self._compute_total_loss(
                y_hat_logits,
                y_targets_for_loss,
                processed_output_states,
                stage_prefix="val",
            )
        except RuntimeError as e:
            self._log_detailed_loss_error(e, y_hat_logits, y_targets_for_loss, "validation")
            raise
        # Log val_loss_total to match JAX backend naming (val_loss/total already logged by _compute_total_loss)
        # Remove redundant val_loss log to avoid duplication
        # self.log("val_loss", ...) removed - use val_loss_total instead

        # Store outputs for epoch-end calculations
        if self.config.training.autoregressive:
            # Store sequence targets for autoregressive accuracy calculation
            outputs = {
                "loss": total_loss,
                "preds": y_hat_logits.detach(),
                "targets": y_targets.detach(),
                "sequence_targets": y_targets_for_loss.detach(),
            }
        else:
            # Store targets aligned with what the loss saw
            outputs = {
                "loss": total_loss,
                "preds": y_hat_logits.detach(),
                "targets": y_targets_for_loss.detach() if isinstance(y_targets_for_loss, torch.Tensor) else y_targets.detach(),
            }

        self._validation_step_outputs.append(outputs)
        return outputs

    def on_validation_epoch_end(self) -> None:
        """Log histograms and bar charts at the end of the validation epoch based on config."""
        if not self._validation_step_outputs:
            return

        preds = torch.cat([x["preds"] for x in self._validation_step_outputs])
        if self.config.training.autoregressive:
            targets = torch.cat([x["sequence_targets"] for x in self._validation_step_outputs])
        else:
            targets = torch.cat([x["targets"] for x in self._validation_step_outputs])

        # Compute and log metrics configured in the YAML file
        # Accept strings or dict specs and normalize them to names (with optional params)
        raw_metrics = self.config.logging.metrics
        normalized: list[tuple[str, str, dict]] = []  # (display_name, internal_name, params)
        if isinstance(raw_metrics, (list, tuple)):
            for item in raw_metrics:
                if isinstance(item, str):
                    normalized.append((item, item, {}))
                elif isinstance(item, dict):
                    if "name" in item and isinstance(item["name"], str):
                        name = item["name"]
                        params = {k: v for k, v in item.items() if k != "name"}
                        normalized.append((name, name, params))
                    elif len(item) == 1:
                        key, val = next(iter(item.items()))
                        if isinstance(key, str):
                            params = val if isinstance(val, dict) else {}
                            normalized.append((key, key, params))
        elif isinstance(raw_metrics, str):
            normalized = [(raw_metrics, raw_metrics, {})]

        # Get paradigm for skipping inapplicable metrics
        paradigm = getattr(self.config.training, "paradigm", "supervised")

        for display_name, metric_name, params in normalized:
            if metric_name in {"loss", "input_seq_len"}:
                continue

            metric_to_compute = metric_name
            if self.config.training.autoregressive and metric_name == "accuracy":
                metric_to_compute = "autoregressive_accuracy"

            # Skip classification metrics for distillation (it's regression, not classification)
            if paradigm == "distillation" and metric_to_compute in {"accuracy", "autoregressive_accuracy", "top_k_accuracy"}:
                continue
            if paradigm == "distillation" and metric_to_compute.startswith("top_"):
                continue

            # Skip top-k metrics in autoregressive aggregation to avoid shape mismatches
            if preds.dim() == 3 and (metric_to_compute.startswith("top_") or metric_to_compute == "top_k_accuracy"):
                continue

            # Handle top-k metrics specified as e.g. "top_5" or dict form {name: 'top_k_accuracy', k: N}
            if metric_to_compute.startswith("top_") and metric_to_compute.split("_")[1].isdigit():
                k_val = int(metric_to_compute.split("_")[1])
                metric_fn = METRICS_REGISTRY.get("top_k_accuracy")
                if metric_fn is None:
                    continue
                metric_val = metric_fn(preds, targets, k=k_val)
            elif metric_to_compute == "top_k_accuracy":
                k_val = int(params.get("k", 5))
                metric_fn = METRICS_REGISTRY.get("top_k_accuracy")
                if metric_fn is None:
                    continue
                metric_val = metric_fn(preds, targets, k=k_val)
            else:
                metric_fn = METRICS_REGISTRY.get(metric_to_compute)
                if metric_fn is None:
                    continue
                metric_val = metric_fn(preds, targets)

            self.log(
                f"val_{display_name}",
                metric_val,
                on_epoch=True,
                prog_bar=(display_name == "accuracy"),
                sync_dist=True,
            )

        # Probing visualizations are handled by dedicated callbacks.

    def on_training_epoch_start(self) -> None:
        """Clear training step outputs at epoch start."""
        self._training_step_outputs.clear()

    def on_training_epoch_end(self) -> None:
        """Compute and log training metrics at epoch end (match JAX backend behavior)."""
        if not self._training_step_outputs:
            return

        preds = torch.cat([x["preds"] for x in self._training_step_outputs])
        targets = torch.cat([x["targets"] for x in self._training_step_outputs])

        # Compute and log metrics configured in the YAML file (same as validation)
        raw_metrics = self.config.logging.metrics
        normalized: list[tuple[str, str, dict]] = []  # (display_name, internal_name, params)
        if isinstance(raw_metrics, (list, tuple)):
            for item in raw_metrics:
                if isinstance(item, str):
                    normalized.append((item, item, {}))
                elif isinstance(item, dict):
                    if "name" in item and isinstance(item["name"], str):
                        name = item["name"]
                        params = {k: v for k, v in item.items() if k != "name"}
                        normalized.append((name, name, params))
                    elif len(item) == 1:
                        key, val = next(iter(item.items()))
                        if isinstance(key, str):
                            params = val if isinstance(val, dict) else {}
                            normalized.append((key, key, params))
        elif isinstance(raw_metrics, str):
            normalized = [(raw_metrics, raw_metrics, {})]

        # Get paradigm for skipping inapplicable metrics
        paradigm = getattr(self.config.training, "paradigm", "supervised")

        for display_name, metric_name, params in normalized:
            if metric_name in {"loss", "input_seq_len"}:
                continue

            metric_to_compute = metric_name
            if self.config.training.autoregressive and metric_name == "accuracy":
                metric_to_compute = "autoregressive_accuracy"

            # Skip classification metrics for distillation (it's regression, not classification)
            if paradigm == "distillation" and metric_to_compute in {"accuracy", "autoregressive_accuracy", "top_k_accuracy"}:
                continue
            if paradigm == "distillation" and metric_to_compute.startswith("top_"):
                continue

            # Skip top-k metrics in autoregressive aggregation to avoid shape mismatches
            if preds.dim() == 3 and (metric_to_compute.startswith("top_") or metric_to_compute == "top_k_accuracy"):
                continue

            # Handle top-k metrics specified as e.g. "top_5" or dict form {name: 'top_k_accuracy', k: N}
            if metric_to_compute.startswith("top_") and metric_to_compute.split("_")[1].isdigit():
                k_val = int(metric_to_compute.split("_")[1])
                metric_fn = METRICS_REGISTRY.get("top_k_accuracy")
                if metric_fn is None:
                    continue
                metric_val = metric_fn(preds, targets, k=k_val)
            elif metric_to_compute == "top_k_accuracy":
                k_val = int(params.get("k", 5))
                metric_fn = METRICS_REGISTRY.get("top_k_accuracy")
                if metric_fn is None:
                    continue
                metric_val = metric_fn(preds, targets, k=k_val)
            else:
                metric_fn = METRICS_REGISTRY.get(metric_to_compute)
                if metric_fn is None:
                    continue
                metric_val = metric_fn(preds, targets)

            # Log training metrics with _epoch suffix to match JAX backend
            self.log(
                f"train_{display_name}",
                metric_val,
                on_epoch=True,
                prog_bar=False,  # Don't clutter progress bar with training metrics
                sync_dist=True,
            )

    def test_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> dict[str, torch.Tensor]:
        """Perform a test step.

        Args:
            batch: Tuple of (inputs, targets)
            batch_idx: Index of the current batch

        Returns:
            Dictionary containing loss and other metrics

        """
        x, y_targets = batch
        y_hat_logits, final_state, _all_states = self(x)

        if self.config.training.autoregressive:
            from soen_toolkit.training.utils.helpers import (
                build_autoregressive_targets,
                extract_token_sequence_from_inputs,
            )

            mode = getattr(self.config.training, "autoregressive_mode", "next_token")
            input_sequence = extract_token_sequence_from_inputs(x)
            y_targets_for_loss = build_autoregressive_targets(input_sequence, y_targets, mode=mode)
        else:
            # Mirror training/validation logic: honor mapping and seq2seq targets
            paradigm = getattr(self.config.training, "paradigm", "supervised")
            mapping = str(getattr(self.config.training, "mapping", "seq2static")).lower()
            if paradigm in ["unsupervised", "self_supervised"]:
                if mapping == "seq2seq":
                    # For seq2seq unsupervised, use sequence outputs to reconstruct input sequence
                    y_targets_for_loss = x
                    # final_state includes initial t=0; drop it to match input length
                    y_hat_logits = final_state[:, 1:, :]
                else:
                    # For seq2static unsupervised, use provided targets (should be summary of inputs)
                    y_targets_for_loss = y_targets
                    # y_hat_logits already set to pooled outputs
            elif isinstance(y_targets, torch.Tensor) and (y_targets.dim() == 3 or (y_targets.dim() == 2 and mapping == "seq2seq")):
                # Sequence targets: align outputs to full sequence and drop initial state
                y_targets_for_loss = y_targets
                y_hat_logits = final_state[:, 1:, :]
            else:
                y_targets_for_loss = y_targets

        processed_states = self.latest_processed_state
        if processed_states is None:
            processed_states = y_hat_logits

        total_loss = self._compute_total_loss(
            y_hat_logits,
            y_targets_for_loss,
            processed_states,
            stage_prefix="test",
        )

        # Store outputs for epoch-end test metric calculations (mirrors validation behavior)
        if self.config.training.autoregressive:
            outputs = {
                "loss": total_loss,
                "preds": y_hat_logits.detach(),
                "targets": y_targets.detach(),
                "sequence_targets": y_targets_for_loss.detach(),
            }
        else:
            outputs = {
                "loss": total_loss,
                "preds": y_hat_logits.detach(),
                # Store the exact targets used for loss to keep shapes aligned for metrics
                "targets": y_targets_for_loss.detach() if isinstance(y_targets_for_loss, torch.Tensor) else y_targets.detach(),
            }
        self._test_step_outputs.append(outputs)

        return {"test_loss": total_loss, "loss": total_loss}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer) -> None:
        """Override to use set_to_none=True for faster gradient zeroing."""
        optimizer.zero_grad(set_to_none=True)

    def configure_optimizers(self):
        """Configure optimizers for training.

        Returns:
            Optimizer: Configured optimizer based on configuration

        """
        # Get optimizer configuration
        optimizer_config = self.config.training.optimizer

        # Handle both dict and OptimizerConfig object formats
        if isinstance(optimizer_config, dict):
            optimizer_name = optimizer_config.get("name", "adamw").lower()
            lr = optimizer_config.get("lr", 0.001)
            optimizer_kwargs = optimizer_config.get("kwargs", {"weight_decay": 1e-4})
        else:
            optimizer_name = optimizer_config.name.lower()
            lr = optimizer_config.lr
            optimizer_kwargs = optimizer_config.kwargs

        # --- Robust numeric conversion for optimizer parameters ---
        from soen_toolkit.training.utils.helpers import (
            safe_convert_optimizer_kwargs,
            safe_numeric_convert,
        )

        # Convert learning rate safely
        lr = safe_numeric_convert(lr, float, "learning_rate")

        # Convert all optimizer kwargs safely
        optimizer_kwargs = safe_convert_optimizer_kwargs(optimizer_kwargs)

        # --- Build parameter list or param groups (name-based) ---
        all_named_params = [(n, p) for n, p in self.named_parameters() if p.requires_grad and torch.is_tensor(p.data)]
        if len(all_named_params) == 0:
            msg = "No valid trainable torch.Tensor parameters found for optimizer configuration. Check model parameters."
            raise RuntimeError(
                msg,
            )

        param_groups_cfg = getattr(optimizer_config, "param_groups", None)
        parameters: list[dict] | list[torch.nn.Parameter]
        if param_groups_cfg:
            import re

            used = set()
            groups: list[dict] = []

            # Helper to normalize a param name to a user-friendly matchable form
            # We strip the common 'model.' prefix for convenience
            def norm(name: str) -> str:
                return name.removeprefix("model.")

            for group in param_groups_cfg:
                if not isinstance(group, dict):
                    continue
                match = group.get("match")
                match_regex = group.get("match_regex")
                # Clean and numeric-convert group optimizer options
                raw_group_opts = {k: v for k, v in group.items() if k not in {"match", "match_regex"}}
                group_opts = safe_convert_optimizer_kwargs(raw_group_opts)
                if "lr" in raw_group_opts:
                    # Ensure LR is a float even if YAML parsed it as a string
                    group_opts["lr"] = safe_numeric_convert(raw_group_opts["lr"], float, "param_group.lr")
                matched_params = []
                for idx, (name, param) in enumerate(all_named_params):
                    if idx in used:
                        continue
                    nname = norm(name)
                    ok = False
                    if isinstance(match, str) and match in nname:
                        ok = True
                    if not ok and isinstance(match_regex, str):
                        try:
                            if re.search(match_regex, nname):
                                ok = True
                        except re.error:
                            pass
                    if ok:
                        matched_params.append(param)
                        used.add(idx)
                if matched_params:
                    group_dict = {"params": matched_params}
                    group_dict.update(group_opts)
                    # Preserve each group's initial LR for scheduler scaling logic
                    group_dict.setdefault("initial_lr", group_dict.get("lr", lr))
                    groups.append(group_dict)

            # Add a fallback group for any remaining params
            remaining = [p for i, (_, p) in enumerate(all_named_params) if i not in used]
            if remaining:
                groups.append({"params": remaining, "lr": lr, "initial_lr": lr})
            parameters = groups
        else:
            parameters = [p for _, p in all_named_params]

        # Create optimizer
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(parameters, lr=lr, **optimizer_kwargs)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(parameters, lr=lr, **optimizer_kwargs)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(parameters, lr=lr, **optimizer_kwargs)
        elif optimizer_name == "rmsprop":
            optimizer = torch.optim.RMSprop(parameters, lr=lr, **optimizer_kwargs)
        elif optimizer_name == "muon":
            # Matrix-aware Muon support: prefer flash-muon (no distributed),
            # otherwise fall back to reference muon with a local 1-process group.
            muon_params = [p for _, p in all_named_params if p.requires_grad and getattr(p, "ndim", 0) >= 2]
            aux_params = [p for _, p in all_named_params if p.requires_grad and getattr(p, "ndim", 0) < 2]

            if len(muon_params) == 0:
                logger.warning("Muon requested but no >=2D parameters found; falling back to AdamW.")
                optimizer = torch.optim.AdamW([p for _, p in all_named_params], lr=lr, **optimizer_kwargs)
            else:
                # Simplified: use global LR for both groups; ignore advanced overrides
                optimizer_kwargs.pop("muon_lr", None)
                optimizer_kwargs.pop("aux_lr", None)
                muon_lr = lr
                aux_lr = lr
                muon_momentum = optimizer_kwargs.pop("momentum", None)
                aux_betas = optimizer_kwargs.pop("betas", None)
                wd = optimizer_kwargs.pop("weight_decay", None)

                # Use reference MuonWithAuxAdam for single-optimizer behavior (scheduler-friendly)
                try:
                    from muon import MuonWithAuxAdam
                    import torch.distributed as dist
                except ModuleNotFoundError as exc:
                    msg = "Muon optimizer requested but the 'muon' package is not installed. Install it with `pip install git+https://github.com/KellerJordan/Muon`."
                    raise ModuleNotFoundError(
                        msg,
                    ) from exc

                try:
                    if dist.is_available() and not dist.is_initialized():
                        tmp_file = Path(tempfile.gettempdir()) / f"soen_muon_store_{uuid4().hex}"
                        dist.init_process_group(
                            backend="gloo",
                            init_method=f"file://{tmp_file}",
                            rank=0,
                            world_size=1,
                        )
                        self._created_local_pg = True
                except Exception as e:
                    logger.warning(f"Failed to init local torch.distributed group for Muon ({e}); falling back to AdamW.")
                    return torch.optim.AdamW([p for _, p in all_named_params], lr=lr, **({"weight_decay": wd} if wd is not None else {}))

                muon_group = {"params": muon_params, "use_muon": True, "lr": muon_lr}
                aux_group = {"params": aux_params, "use_muon": False, "lr": aux_lr}
                if wd is not None:
                    muon_group["weight_decay"] = wd
                    aux_group["weight_decay"] = wd
                # Keep defaults for momentum/betas unless explicitly set
                if muon_momentum is not None:
                    muon_group["momentum"] = muon_momentum
                if aux_betas is not None:
                    aux_group["betas"] = aux_betas
                optimizer = MuonWithAuxAdam([muon_group, aux_group])
        elif optimizer_name == "lion":
            try:
                from lion_pytorch import Lion
            except ModuleNotFoundError as exc:
                msg = "Lion optimizer requested but the 'lion-pytorch' package is not installed. Install it with `pip install lion-pytorch` to proceed."
                raise ModuleNotFoundError(
                    msg,
                ) from exc
            optimizer = Lion(parameters, lr=lr, **optimizer_kwargs)
        else:
            msg = f"Unsupported optimizer: {optimizer_name}"
            raise ValueError(msg)

        return optimizer

    def on_validation_epoch_start(self) -> None:
        """Reset stored sample output and label at the beginning of each validation epoch."""
        super().on_validation_epoch_start()

        self._validation_step_outputs.clear()

    def on_test_epoch_start(self) -> None:
        """Reset stored outputs at the beginning of the test epoch."""
        super().on_test_epoch_start()
        self._test_step_outputs.clear()

    def on_test_epoch_end(self) -> None:
        """Compute and log configured test metrics at epoch end so they appear in PL's test summary table."""
        if not self._test_step_outputs:
            return

        preds = torch.cat([x["preds"] for x in self._test_step_outputs])
        if self.config.training.autoregressive:
            targets = torch.cat([x["sequence_targets"] for x in self._test_step_outputs])
        else:
            targets = torch.cat([x["targets"] for x in self._test_step_outputs])

        # Normalize metrics from config (same as validation)
        raw_metrics = self.config.logging.metrics
        normalized: list[tuple[str, str, dict]] = []  # (display_name, internal_name, params)
        if isinstance(raw_metrics, (list, tuple)):
            for item in raw_metrics:
                if isinstance(item, str):
                    normalized.append((item, item, {}))
                elif isinstance(item, dict):
                    if "name" in item and isinstance(item["name"], str):
                        name = item["name"]
                        params = {k: v for k, v in item.items() if k != "name"}
                        normalized.append((name, name, params))
                    elif len(item) == 1:
                        key, val = next(iter(item.items()))
                        if isinstance(key, str):
                            params = val if isinstance(val, dict) else {}
                            normalized.append((key, key, params))
        elif isinstance(raw_metrics, str):
            normalized = [(raw_metrics, raw_metrics, {})]

        for display_name, metric_name, params in normalized:
            if metric_name in {"loss", "input_seq_len"}:
                continue

            metric_to_compute = metric_name
            if self.config.training.autoregressive and metric_name == "accuracy":
                metric_to_compute = "autoregressive_accuracy"

            # Skip top-k metrics for AR aggregation
            if preds.dim() == 3 and (metric_to_compute.startswith("top_") or metric_to_compute == "top_k_accuracy"):
                continue

            if metric_to_compute.startswith("top_") and metric_to_compute.split("_")[1].isdigit():
                k_val = int(metric_to_compute.split("_")[1])
                metric_fn = METRICS_REGISTRY.get("top_k_accuracy")
                if metric_fn is None:
                    continue
                metric_val = metric_fn(preds, targets, k=k_val)
            elif metric_to_compute == "top_k_accuracy":
                k_val = int(params.get("k", 5))
                metric_fn = METRICS_REGISTRY.get("top_k_accuracy")
                if metric_fn is None:
                    continue
                metric_val = metric_fn(preds, targets, k=k_val)
            else:
                metric_fn = METRICS_REGISTRY.get(metric_to_compute)
                if metric_fn is None:
                    continue
                metric_val = metric_fn(preds, targets)

            self.log(
                f"test_{display_name}",
                metric_val,
                on_epoch=True,
                prog_bar=(display_name == "accuracy"),
                sync_dist=True,
            )

    # ---------------- Constraint enforcement hooks ----------------
    def on_before_zero_grad(self, optimizer) -> None:
        """Clamp parameters to configured bounds immediately after optimizer.step()."""
        try:
            if hasattr(self, "model") and hasattr(self.model, "enforce_param_constraints"):
                self.model.enforce_param_constraints()
        except Exception as e:
            logger.warning(f"Failed to enforce parameter constraints in on_before_zero_grad: {e}")

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """Ensure constraints/masks are applied before checkpoint serialization."""
        try:
            if hasattr(self, "model") and hasattr(self.model, "enforce_param_constraints"):
                self.model.enforce_param_constraints()
        except Exception as e:
            logger.warning(f"Failed to enforce parameter constraints before saving checkpoint: {e}")

    def _validate_model_output_dimension(self, soen_model: SOENModelCore) -> None:
        """Validate that model output dimension matches dataset num_classes for classification tasks.

        Args:
            soen_model: The SOEN model core to validate

        Raises:
            ValueError: If model output dimension doesn't match dataset num_classes
        """
        # Only validate for classification tasks
        if not getattr(self.config.training, "classification", False):
            return

        # Get dataset num_classes from config
        num_classes = getattr(self.config.data, "num_classes", None)
        if num_classes is None:
            return

        # Get model output dimension (last layer's dimension)
        if not hasattr(soen_model, "layer_nodes") or not soen_model.layer_nodes:
            logger.warning("Cannot validate output dimension: model.layer_nodes not available")
            return

        # Get the last layer's dimension
        sorted_layer_ids = sorted(soen_model.layer_nodes.keys())
        if not sorted_layer_ids:
            logger.warning("Cannot validate output dimension: no layers found")
            return

        last_layer_id = sorted_layer_ids[-1]
        model_output_dim = soen_model.layer_nodes[last_layer_id]

        # Compare with dataset num_classes
        if model_output_dim != num_classes:
            paradigm = getattr(self.config.training, "paradigm", "supervised")
            mapping = getattr(self.config.training, "mapping", "seq2static")

            error_msg = [
                "=" * 80,
                "MODEL OUTPUT DIMENSION MISMATCH DETECTED",
                "=" * 80,
                "",
                f"Your model's output layer has dimension {model_output_dim}, but your dataset",
                f"expects {num_classes} classes.",
                "",
                "This mismatch will cause training to fail with errors like:",
                f"  - PyTorch: 'IndexError: Target {num_classes - 1} is out of bounds'",
                "  - JAX: NaN losses or incorrect predictions",
                "",
                "DETAILS:",
                f"  • Model output layer (layer_id={last_layer_id}): {model_output_dim} dimensions",
                f"  • Dataset num_classes (config.data.num_classes): {num_classes}",
                f"  • Task: {paradigm} classification ({mapping})",
                "",
                "HOW TO FIX:",
                "",
                "Option 1: Rebuild your model with the correct output dimension",
                f"  - Modify your model YAML/architecture to set the last layer's 'dim' to {num_classes}",
                f"  - Or rebuild from model_creation_gui with output_dim={num_classes}",
                "",
                "Option 2: Regenerate your dataset to match the model",
                f"  - If your dataset generator supports it, set num_classes={model_output_dim}",
                "  - Or exclude classes that don't fit (e.g., set include_space=False if model has 26 outputs)",
                "",
                "Option 3: Check your dataset generation script",
                "  - Ensure dataset labels use values in range [0, num_classes-1]",
                "  - For seq2seq tasks: padding should use -100 (already handled by framework)",
                "",
                "=" * 80,
            ]

            logger.error("\n".join(error_msg))
            raise ValueError(f"Model output dimension ({model_output_dim}) does not match dataset num_classes ({num_classes}). See error message above for detailed fix instructions.")

    def _log_detailed_loss_error(self, error: Exception, outputs, targets, stage: str) -> None:
        """Log detailed error message with configuration suggestions."""
        import logging

        error_logger = logging.getLogger(__name__)

        # Get current config values
        paradigm = getattr(self.config.training, "paradigm", "supervised")
        mapping = getattr(self.config.training, "mapping", "seq2static")

        # Get shapes for analysis
        output_shape = tuple(outputs.shape) if isinstance(outputs, torch.Tensor) else str(type(outputs))
        target_shape = tuple(targets.shape) if isinstance(targets, torch.Tensor) else str(type(targets))

        error_msg = [
            f"[ERROR] {stage.title()} loss computation failed!",
            f"   Output shape: {output_shape}",
            f"   Target shape: {target_shape}",
            f"   Current config: paradigm='{paradigm}', mapping='{mapping}'",
            f"   Error: {error!s}",
        ]

        # Analyze error and provide specific suggestions
        suggestions = self._analyze_shape_mismatch(output_shape, target_shape, paradigm, mapping, error)
        if suggestions:
            error_msg.extend(["", "[SUGGESTIONS]"] + [f"   • {s}" for s in suggestions])

        error_logger.error("\n".join(error_msg))

    def _analyze_shape_mismatch(self, output_shape, target_shape, paradigm, mapping, error):
        """Analyze shape mismatch and provide specific suggestions."""
        suggestions = []
        error_str = str(error).lower()

        # Handle common error patterns
        if "expected target size" in error_str and "got" in error_str:
            # Cross-entropy dimension mismatch
            if isinstance(output_shape, tuple) and isinstance(target_shape, tuple):
                if len(output_shape) == 3 and len(target_shape) == 2:
                    # seq2seq classification: [B, T, C] vs [B, T]
                    suggestions.append(f"Set mapping='seq2seq' for sequence classification (current: '{mapping}')")
                    suggestions.append("Your targets have shape [N, T] suggesting per-timestep classification")
                elif len(output_shape) == 2 and len(target_shape) == 2 and output_shape[1] != target_shape[1]:
                    # seq2static classification: output [B, C] vs target [B, T]
                    suggestions.append(f"For sequence classification, set mapping='seq2seq' (current: '{mapping}')")
                    suggestions.append("Or use seq2static with targets shape [N] for whole-sequence classification")

        elif "size of tensor" in error_str and "must match" in error_str:
            # Broadcasting/size mismatch (typically regression)
            if isinstance(output_shape, tuple) and isinstance(target_shape, tuple):
                if len(output_shape) == 2 and len(target_shape) == 3:
                    # seq2static output vs seq2seq target
                    suggestions.append("Output suggests seq2static but targets suggest seq2seq")
                    suggestions.append("Set mapping='seq2seq' if you want sequence-to-sequence prediction")
                    suggestions.append("Or reshape targets to [N, K] for seq2static regression")
                elif len(output_shape) == 3 and len(target_shape) == 2:
                    if target_shape[1] != output_shape[1]:
                        # Sequence length mismatch
                        suggestions.append(f"Sequence length mismatch: output T={output_shape[1]}, target T={target_shape[1]}")
                        suggestions.append(f"Set data.target_seq_len={target_shape[1]} to match your targets")

        elif "0d or 1d target tensor expected" in error_str:
            # seq2seq classification with wrong cross_entropy usage
            suggestions.append("For sequence classification, targets should be 2D [N, T], not 3D")
            suggestions.append("The cross_entropy loss has been updated to handle seq2seq automatically")
            suggestions.append("Check that your targets have integer dtype and shape [N, T]")

        # Paradigm-specific suggestions
        if paradigm == "supervised" and mapping == "seq2static":
            suggestions.append("For seq2static: use targets shape [N] (classification) or [N, K] (regression)")
        elif paradigm == "supervised" and mapping == "seq2seq":
            suggestions.append("For seq2seq: use targets shape [N, T] (classification) or [N, T, K] (regression)")
        elif paradigm in ["unsupervised", "self_supervised"]:
            if mapping == "seq2seq":
                suggestions.append("Self-supervised seq2seq should reconstruct input sequences")
            else:
                suggestions.append("Self-supervised seq2static should predict sequence summaries")

        # Data configuration suggestions
        if not hasattr(self.config.data, "target_seq_len") or not self.config.data.target_seq_len:
            suggestions.append("Set data.target_seq_len to control sequence length alignment")

        return suggestions
