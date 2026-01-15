from typing import Any

import torch


def safe_numeric_convert(value: Any, target_type: type = float, param_name: str = "parameter") -> Any:
    """Safely convert a value to a numeric type, handling YAML parsing edge cases.

    This function handles cases where YAML parsers might interpret scientific notation
    (like 1e-4) as strings instead of floats, or where other numeric formats need
    conversion.

    Args:
        value: The value to convert
        target_type: The target numeric type (float, int)
        param_name: Name of the parameter for error messages

    Returns:
        Converted numeric value, or original value if not convertible

    Raises:
        ValueError: If the value should be numeric but cannot be converted

    """
    # If already the correct type, return as-is
    if isinstance(value, target_type):
        return value

    # If it's a string, try to convert it
    if isinstance(value, str):
        # Handle common scientific notation formats
        try:
            if target_type is float:
                return float(value)
            if target_type is int:
                # For int conversion, handle float strings too
                return int(float(value))
            return target_type(value)
        except (ValueError, TypeError) as e:
            msg = f"Parameter '{param_name}' with value '{value}' cannot be converted to {target_type.__name__}. Expected a numeric value but got: {type(value).__name__}. Original error: {e}"
            raise ValueError(
                msg,
            )

    # If it's another numeric type, convert it
    if isinstance(value, (int, float, complex)):
        try:
            return target_type(value)
        except (ValueError, TypeError, OverflowError) as e:
            msg = f"Parameter '{param_name}' with value '{value}' cannot be converted to {target_type.__name__}. Original error: {e}"
            raise ValueError(
                msg,
            )

    # For non-numeric types, return as-is (might be intentional, like None or bool)
    return value


def safe_convert_optimizer_kwargs(optimizer_kwargs: dict[str, Any]) -> dict[str, Any]:
    """Safely convert optimizer kwargs to appropriate numeric types.

    This handles common optimizer parameters that should be numeric but might
    be parsed as strings from YAML files.

    Args:
        optimizer_kwargs: Dictionary of optimizer keyword arguments

    Returns:
        Dictionary with properly converted numeric values

    """
    # Common optimizer parameters that should be floats
    float_params = {
        "weight_decay",
        "eps",
        "momentum",
        "dampening",
        "alpha",
        "rho",
        "lr_decay",
        "initial_accumulator_value",
        "lambd",
        "beta",
        "centered",
        "amsgrad",  # Some optimizers use these
    }

    # Parameters that should be integers
    int_params = {
        "maximize",  # Some optimizers have integer flags
    }

    # Parameters that are typically lists/tuples of floats
    float_list_params = {
        "betas",  # Adam/AdamW betas parameter
    }

    # Parameters that should remain as booleans
    bool_params = {
        "amsgrad",
        "maximize",
        "nesterov",
        "centered",
    }

    cleaned_kwargs: dict[str, Any] = {}

    for key, value in optimizer_kwargs.items():
        try:
            if key in bool_params:
                # Handle boolean parameters
                if isinstance(value, str):
                    cleaned_kwargs[key] = value.lower() in ("true", "1", "yes", "on")
                else:
                    cleaned_kwargs[key] = bool(value)

            elif key in int_params:
                # Handle integer parameters
                cleaned_kwargs[key] = safe_numeric_convert(value, int, key)

            elif key in float_params:
                # Handle float parameters
                cleaned_kwargs[key] = safe_numeric_convert(value, float, key)

            elif key in float_list_params:
                # Handle list/tuple of floats (like betas)
                if isinstance(value, (list, tuple)):
                    cleaned_kwargs[key] = [safe_numeric_convert(v, float, f"{key}[{i}]") for i, v in enumerate(value)]
                else:
                    # Single value that should be a list
                    cleaned_kwargs[key] = [safe_numeric_convert(value, float, key)]

            # For unknown parameters, try to convert if it's a string that looks numeric
            elif isinstance(value, str):
                try:
                    # Try float first (most common case)
                    cleaned_kwargs[key] = float(value)
                except ValueError:
                    # If that fails, keep as string
                    cleaned_kwargs[key] = value
            else:
                cleaned_kwargs[key] = value

        except ValueError as e:
            # Re-raise with more context
            msg = f"Error processing optimizer parameter '{key}': {e}"
            raise ValueError(msg)

    return cleaned_kwargs


def safe_convert_scheduler_params(scheduler_params: dict[str, Any]) -> dict[str, Any]:
    """Safely convert scheduler parameters to appropriate numeric types.

    Args:
        scheduler_params: Dictionary of scheduler parameters

    Returns:
        Dictionary with properly converted numeric values

    """
    # Common scheduler parameters that should be floats
    float_params = {
        "max_lr",
        "min_lr",
        "lr",
        "warmup_start_lr",
        "factor_increase",
        "factor_decrease",
        "increase_factor",
        "decrease_factor",
        "threshold",
        "restart_decay",
        "period_decay",
        "amplitude_decay",
        "ema_beta",
        "min_delta",
    }

    # Parameters that should be integers
    int_params = {
        "warmup_epochs",
        "cycle_epochs",
        "patience",
        "patience_increase",
        "patience_decrease",
        "cooldown",
        "batches_per_adjustment",
        "adjustment_frequency",
        "total_steps",
        "start_epoch",
        "step_size",
        "n",
    }

    # Parameters that should remain as booleans
    bool_params = {
        "enable_restarts",
        "adjust_on_batch",
        "soft_restart",
        "debug",
        "log_space",
        "intra_epoch",
        "enabled",
        "verbose",
        "active",
    }

    # Parameters that should remain as strings
    string_params = {
        "type",
        "monitor_metric",
        "mode",
        "threshold_mode",
        "growth_type",
        "name",
    }

    cleaned_params: dict[str, Any] = {}

    for key, value in scheduler_params.items():
        try:
            if key in bool_params:
                if isinstance(value, str):
                    cleaned_params[key] = value.lower() in ("true", "1", "yes", "on")
                else:
                    cleaned_params[key] = bool(value)

            elif key in int_params:
                cleaned_params[key] = safe_numeric_convert(value, int, key)

            elif key in float_params:
                cleaned_params[key] = safe_numeric_convert(value, float, key)

            elif key in string_params:
                cleaned_params[key] = str(value)

            # For nested dictionaries (like warmup config), recurse
            elif isinstance(value, dict):
                cleaned_params[key] = safe_convert_scheduler_params(value)
            elif isinstance(value, str):
                # Try to convert unknown string parameters
                try:
                    cleaned_params[key] = float(value)
                except ValueError:
                    cleaned_params[key] = value
            else:
                cleaned_params[key] = value

        except ValueError as e:
            msg = f"Error processing scheduler parameter '{key}': {e}"
            raise ValueError(msg)

    return cleaned_params


# -----------------------------------------------------------------------------
# Autoregressive utilities
# -----------------------------------------------------------------------------


def extract_token_sequence_from_inputs(inputs: torch.Tensor) -> torch.Tensor:
    """Derive integer token indices from model inputs for autoregressive training.

    Rules:
    - If inputs are one-hot encoded [batch, seq_len, vocab], take argmax over vocab
    - Else assume inputs are already integer indices
    cast to long
    """
    if inputs.dim() == 3 and inputs.shape[-1] > 1:
        return inputs.argmax(dim=-1)
    return inputs.long()


def build_autoregressive_targets(
    input_sequence: torch.Tensor,
    final_label: torch.Tensor,
    mode: str = "next_token",
) -> torch.Tensor:
    """Construct autoregressive target sequence.

    For autoregressive training, targets are derived from the input sequence itself,
    not from external labels. The goal is next-token prediction: predict token t+1
    given tokens 0..t.

    Args:
        input_sequence: Token indices [batch, seq_len] extracted from inputs
        final_label: Ignored for AR training (kept for API compatibility)
        mode: "next_token" (standard AR) or "seq2seq" (use provided labels)

    Returns:
        Target sequence [batch, seq_len] where targets[i] = input_sequence[i+1]

    Examples:
        Input:  [t, h, e, _, c, a, t]
        Target: [h, e, _, c, a, t, <last_token>]

        For the last position, we use the last token from input_sequence
        (this means the model learns to predict the final token given all previous)
    """
    if mode == "next_token":
        # Standard autoregressive: shift input sequence by 1
        # Target at position i is the input at position i+1
        # For the last position, use the last input token (model predicts end-of-sequence)
        batch_size, seq_len = input_sequence.shape

        # Shift: take tokens 1 through end
        shifted = input_sequence[:, 1:]  # [batch, seq_len-1]

        # For the last position, use the last token from the input
        # This teaches the model to predict the sequence end
        last_token = input_sequence[:, -1:]  # [batch, 1]

        # Concatenate to get full target sequence
        targets = torch.cat([shifted, last_token], dim=1)  # [batch, seq_len]

        return targets

    elif mode == "seq2seq":
        # Use provided labels (for non-AR seq2seq tasks)
        # Handle various label shapes
        if final_label.dim() == 1:
            # [batch] - single label per sequence, not suitable for seq2seq
            raise ValueError(
                "seq2seq mode requires sequence labels [batch, seq_len], got [batch] shape. "
                "For autoregressive training, use mode='next_token' instead."
            )
        elif final_label.dim() == 2:
            # [batch, seq_len] - perfect
            return final_label
        elif final_label.dim() == 3:
            # [batch, seq_len, 1] or [batch, seq_len, features] - squeeze if needed
            if final_label.shape[-1] == 1:
                return final_label.squeeze(-1)
            else:
                raise ValueError(
                    f"seq2seq mode expects token indices [batch, seq_len], "
                    f"got shape {final_label.shape}. Last dimension should be 1 or omitted."
                )
        else:
            raise ValueError(
                f"Unexpected label shape for seq2seq mode: {final_label.shape}. "
                f"Expected [batch, seq_len]."
            )
    else:
        msg = f"Unknown autoregressive mode: '{mode}'. Valid modes: 'next_token', 'seq2seq'"
        raise ValueError(msg)


def standardize_outputs_for_autoregressive(
    outputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Ensure outputs time dimension matches targets by skipping the initial state
    if present (common SOEN behavior: seq_len+1 outputs).
    """
    if outputs.dim() != 3:
        return outputs
    if outputs.size(1) == targets.size(1) + 1:
        return outputs[:, 1:, :]
    return outputs
