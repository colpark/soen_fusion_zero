#!/usr/bin/env python3
"""Core metric computation functions for criticality optimization.

These functions were moved from optuna_runner.py to support the modular metrics system.
"""

import math
from typing import Any

import numpy as np
import torch

from soen_toolkit.core import SOENModelCore


def _layer_ids(model: SOENModelCore) -> list[int]:
    return [cfg.layer_id for cfg in model.layers_config]


def _parse_conn_key(name: str) -> tuple[int, int]:
    parts = name.split("_")
    return int(parts[1]), int(parts[3])


def _phi_offset_vec(model: SOENModelCore, lid: int) -> torch.Tensor | None:
    lids = _layer_ids(model)
    layer = model.layers[lids.index(lid)]
    phi_off = getattr(layer, "phi_offset", None)
    if phi_off is None:
        return None
    if isinstance(phi_off, torch.nn.Parameter):
        return phi_off.data
    try:
        return torch.as_tensor(phi_off, device=next(layer.parameters(), torch.zeros(1)).device)
    except Exception:
        return None


def compute_phi_in_all_layers_adjusted(
    model: SOENModelCore,
    s_histories: list[torch.Tensor],
    *,
    subtract_phi_offset: bool = True,
) -> dict[int, torch.Tensor]:
    lids = _layer_ids(model)
    B = s_histories[0].shape[0]
    T = s_histories[0].shape[1] - 1
    phi_in = {lid: torch.zeros(B, T, s_histories[lids.index(lid)].shape[-1], device=s_histories[0].device) for lid in lids}

    for name, J in model.connections.items():
        if not name.startswith("J_"):
            continue
        from_l, to_l = _parse_conn_key(name)
        s_src = s_histories[lids.index(from_l)][:, 1:, :]  # [B,T,D_src]
        phi_in[to_l] = phi_in[to_l] + torch.matmul(s_src, J.t())  # [B,T,D_to]

    if subtract_phi_offset:
        for lid in lids:
            off = _phi_offset_vec(model, lid)
            if off is not None:
                phi_in[lid] = phi_in[lid] - off.view(1, 1, -1)
    return phi_in


def precompute_out_sums(model):
    out_sum_by_from = {}
    for name, J in model.connections.items():
        if not name.startswith("J_"):
            continue
        f, _t = _parse_conn_key(name)
        v = torch.sum(J, dim=0)
        out_sum_by_from[f] = out_sum_by_from.get(f, 0) + v
    return out_sum_by_from


def _build_per_node_connectivity_masks(
    model: SOENModelCore,
    s_histories: list[torch.Tensor],
) -> tuple[dict[int, torch.Tensor], dict[int, torch.Tensor]]:
    """Build per-node masks indicating which neurons have incoming/outgoing connections.

    Returns:
        has_incoming: dict[layer_id, bool_tensor[D]] - True if neuron has at least one incoming connection
        has_outgoing: dict[layer_id, bool_tensor[D]] - True if neuron has at least one outgoing connection

    Note: phi_offset counts as an incoming connection source for neurons with non-zero offset.
    """
    lids = _layer_ids(model)
    device = s_histories[0].device if s_histories else torch.device("cpu")

    # Initialize masks as all False
    has_incoming: dict[int, torch.Tensor] = {}
    has_outgoing: dict[int, torch.Tensor] = {}

    for lid in lids:
        D = s_histories[lids.index(lid)].shape[-1]
        has_incoming[lid] = torch.zeros(D, dtype=torch.bool, device=device)
        has_outgoing[lid] = torch.zeros(D, dtype=torch.bool, device=device)

    # Parse connections to mark connected neurons
    for name, J in model.connections.items():
        if not name.startswith("J_"):
            continue
        try:
            from_l, to_l = _parse_conn_key(name)
        except Exception:
            continue

        # J shape is [D_to, D_from]
        # A source neuron has outgoing if any row of J for that column is non-zero
        # A target neuron has incoming if any column of J for that row is non-zero
        if from_l in has_outgoing:
            # Check which source neurons have non-zero weights (any row for each column)
            has_outgoing[from_l] |= (J.abs().sum(dim=0) > 1e-12)
        if to_l in has_incoming:
            # Check which target neurons have non-zero weights (any column for each row)
            has_incoming[to_l] |= (J.abs().sum(dim=1) > 1e-12)

    # phi_offset counts as an incoming connection source
    for lid in lids:
        idx = lids.index(lid)
        layer = model.layers[idx]
        phi_off = getattr(layer, "phi_offset", None)
        if phi_off is not None:
            phi_off_t = torch.as_tensor(phi_off, device=device, dtype=torch.float32)
            if phi_off_t.dim() == 0:
                # Scalar phi_offset applies to all neurons
                if phi_off_t.abs().item() > 1e-12:
                    has_incoming[lid] = torch.ones_like(has_incoming[lid])
            else:
                # Per-neuron phi_offset
                has_incoming[lid] |= (phi_off_t.abs() > 1e-12)

    return has_incoming, has_outgoing


def compute_branching_ratio(model, s_histories, target_layers=None, min_threshold=1e-6, center_s=False):
    """Robust branching ratio estimation using least-squares regression.

    Avoids division-by-tiny-numbers by solving: phi_out ≈ σ * phi_in
    For terminal layers (no outgoing connections), uses the layer's states
    themselves as the total outgoing signal (phi_out = s).

    Key improvements:
    - Only includes neurons that have BOTH incoming and outgoing connections
    - Neurons with no connections are excluded from the computation to avoid extreme values
    - phi_offset counts as an input source for neurons
    """
    lids = _layer_ids(model)
    phi_in = compute_phi_in_all_layers_adjusted(model, s_histories, subtract_phi_offset=True)
    targets = target_layers or lids
    out_sum_by_from = precompute_out_sums(model)

    # Build per-node connectivity masks
    has_incoming, has_outgoing = _build_per_node_connectivity_masks(model, s_histories)

    sigmas = []

    for lid in targets:
        s = s_histories[lids.index(lid)][:, 1:, :]  # [B, T, D]
        if center_s:
            s = s - s.mean(dim=1, keepdim=True)

        out_sum_cols = out_sum_by_from.get(lid)
        # Terminal layer handling: if no outgoing connections, use states as phi_out
        phi_out = s if out_sum_cols is None else s * out_sum_cols.view(1, 1, -1)
        phi_in_layer = phi_in.get(lid)
        if phi_in_layer is None:
            continue

        # Build mask for neurons that have BOTH incoming AND outgoing connections
        incoming_mask = has_incoming.get(lid)
        outgoing_mask = has_outgoing.get(lid)
        if incoming_mask is None or outgoing_mask is None:
            continue

        connected_mask = incoming_mask & outgoing_mask  # [D]
        num_connected = connected_mask.sum().item()

        if num_connected == 0:
            # No connected neurons in this layer - skip it
            continue

        # Expand mask for broadcasting: [1, 1, D]
        connected_mask_expanded = connected_mask.view(1, 1, -1)

        # Apply mask to select only connected neurons before flattening
        phi_in_masked = phi_in_layer * connected_mask_expanded.float()
        phi_out_masked = phi_out * connected_mask_expanded.float()

        # Flatten for regression: phi_out ≈ σ * phi_in
        x = phi_in_masked.reshape(-1)  # [B*T*D]
        y = phi_out_masked.reshape(-1)  # [B*T*D]

        # Mask out zero values from disconnected neurons and tiny denominators
        value_mask = x.abs() > min_threshold
        if value_mask.sum() < 10:  # Need minimum data points
            sigmas.append(0.0)  # Default to subcritical if insufficient data
            continue

        x_clean = x[value_mask]
        y_clean = y[value_mask]

        # Robust least squares: σ = (x^T y) / (x^T x)
        numerator = torch.sum(x_clean * y_clean)
        denominator = torch.sum(x_clean * x_clean) + 1e-10  # Small regularization
        sigma_hat = numerator / denominator

        # Clamp to reasonable range (much tighter than before)
        sigma_hat = torch.clamp(sigma_hat, -3.0, 3.0)
        sigmas.append(float(sigma_hat.item()))

    return float(np.mean(sigmas)) if sigmas else float("nan")


def criticality_branching_cost(sigma: float, target: float = 1.0) -> float:
    """Improved branching ratio cost function with better gradients near criticality.

    Uses log-barrier style penalty that provides strong gradients as sigma approaches target,
    with asymmetric treatment (slightly prefer supercritical over subcritical).
    """
    if math.isnan(sigma):
        return 100.0  # Heavy penalty for invalid states

    # Asymmetric cost: slightly prefer supercritical (sigma > 1) over subcritical
    deviation = sigma - target

    if abs(deviation) < 0.01:  # Very close to critical
        # Quadratic basin near target for fine-tuning
        return 0.5 * (deviation / 0.01) ** 2
    if deviation > 0:  # Supercritical (sigma > target)
        # Gentler penalty for supercritical states
        return 0.5 + 2.0 * (deviation - 0.01)
    # Subcritical (sigma < target)
    # Steeper penalty for subcritical states
    return 0.5 + 3.0 * abs(deviation - 0.01)


def psd_energy_series(s_histories: list[torch.Tensor]) -> torch.Tensor:
    Es = []
    for s in s_histories:
        Es.append((s[:, 1:, :] ** 2).sum(dim=-1))  # [B,T]
    E = torch.stack(Es, dim=0).sum(dim=0)  # [B,T]
    return E.mean(dim=0)  # [T]


def welch_psd(x: torch.Tensor, dt_s: float, seg_len: int | None = None, overlap: float = 0.5) -> tuple[torch.Tensor, torch.Tensor]:
    x = x - x.mean()
    T = x.numel()
    nperseg = int(seg_len or min(256, T))
    noverlap = int(nperseg * overlap)
    step = max(1, nperseg - noverlap)
    window = torch.hann_window(nperseg, device=x.device)
    acc = None
    count = 0
    for start in range(0, max(0, T - nperseg + 1), step):
        seg = x[start : start + nperseg]
        segw = seg * window
        X = torch.fft.rfft(segw)
        Pxx = (X.conj() * X).real / (window.pow(2).sum())
        acc = Pxx if acc is None else (acc + Pxx)
        count += 1
    if count == 0:
        f = torch.fft.rfftfreq(T, d=dt_s)
        return f, torch.zeros_like(f)
    Pxx = acc / count
    f = torch.fft.rfftfreq(nperseg, d=dt_s)
    return f, Pxx


def temporal_psd_beta_and_cost(s_histories, dt_s, *, target_beta=2.0, fmin_frac=0.02, fmax_frac=0.3):
    E = psd_energy_series(s_histories)  # [T]
    f, Pxx = welch_psd(E, dt_s)
    if f.numel() < 4:
        return float("nan"), float("nan")
    f_nyq = 0.5 / dt_s
    fmin = fmin_frac * f_nyq
    fmax = fmax_frac * f_nyq
    mask = (f > 0) & (f >= fmin) & (f <= fmax)
    if mask.sum() < 3:
        return float("nan"), float("nan")
    xf = torch.log10(f[mask])
    yf = torch.log10(Pxx[mask] + 1e-20)
    A = torch.stack([xf, torch.ones_like(xf)], dim=1)
    beta = float(-torch.linalg.lstsq(A, yf.unsqueeze(1)).solution[0].item())
    cost = (beta - target_beta) ** 2
    return beta, cost


def _grid_coords(num_nodes: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically place `num_nodes` on a compact 2D integer grid.
    Same as in connectivity.py - places nodes row-major on ceil(sqrt(N)) × ceil(N/cols) grid.
    """
    if num_nodes <= 0:
        return torch.empty(0), torch.empty(0)

    n_cols = math.ceil(math.sqrt(num_nodes))
    math.ceil(num_nodes / n_cols)

    coords_x = []
    coords_y = []

    for i in range(num_nodes):
        row = i // n_cols
        col = i % n_cols
        coords_x.append(float(col))
        coords_y.append(float(row))

    return torch.tensor(coords_x), torch.tensor(coords_y)


def spatial_psd_beta_for_layer(s_hist: torch.Tensor, *, target_beta: float = 2.0) -> tuple[float, float]:
    """Compute spatial PSD using power_law grid coordinates with fallbacks for small layers.

    This improved version works with any layer size by using multiple analysis methods:
    1. Full 2D FFT analysis (for layers with >= 4 neurons)
    2. Distance-based correlation analysis (for small layers with >= 2 neurons)
    3. Variance-based spatial measure (fallback for single neurons)
    """
    # s_hist: [B, T+1, D]
    if s_hist.dim() != 3:
        return float("nan"), float("nan")
    B, Tp1, D = s_hist.shape
    if Tp1 < 2:
        return float("nan"), float("nan")

    T = Tp1 - 1

    # Get grid coordinates using power_law placement
    x_coords, y_coords = _grid_coords(D)
    if len(x_coords) != D:
        return float("nan"), float("nan")

    # Create rectangular grid that fits all points
    x_max = int(x_coords.max().item()) + 1
    y_max = int(y_coords.max().item()) + 1

    # Method 1: Full 2D FFT analysis (for larger layers)
    if D >= 4 and x_max >= 2 and y_max >= 2:
        s = s_hist[:, 1:, :]  # [B, T, D]

        # Map neurons to spatial grid
        spatial_grid = torch.zeros(B, T, y_max, x_max, device=s.device)

        for neuron_idx in range(D):
            x_idx = int(x_coords[neuron_idx].item())
            y_idx = int(y_coords[neuron_idx].item())
            spatial_grid[:, :, y_idx, x_idx] = s[:, :, neuron_idx]

        # Compute 2D FFT
        S = torch.fft.fft2(spatial_grid, dim=(-2, -1))
        S = torch.fft.fftshift(S, dim=(-2, -1))
        P = (S.conj() * S).real.mean(dim=(0, 1))  # Average over batch and time

        # Compute radial average
        yy, xx = torch.meshgrid(torch.arange(y_max, device=s.device), torch.arange(x_max, device=s.device), indexing="ij")
        cy, cx = y_max // 2, x_max // 2
        r = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2).long()
        rmax = int(r.max().item())

        radial, ks = [], []
        for k in range(1, rmax + 1):
            m = r == k
            if m.any():
                radial.append(P[m].mean())
                ks.append(k)

        if len(ks) >= 3:
            ks = torch.tensor(ks, device=s.device, dtype=torch.float32)
            radial = torch.stack(radial)

            # Fit power law in log-log space
            xf = torch.log10(ks)
            yf = torch.log10(radial + 1e-20)
            A = torch.stack([xf, torch.ones_like(xf)], dim=1)
            beta = float(-torch.linalg.lstsq(A, yf.unsqueeze(1)).solution[0].item())

            return beta, (beta - target_beta) ** 2

    # Method 2: Distance-based correlation analysis (for small layers)
    # This analyzes how activity correlates with spatial distance between neurons
    if D >= 2:
        s = s_hist[:, 1:, :]  # [B, T, D]

        # Compute pairwise distances using grid coordinates
        distances = []
        correlations = []

        for i in range(D):
            for j in range(i + 1, D):
                # Spatial distance
                dx = x_coords[i] - x_coords[j]
                dy = y_coords[i] - y_coords[j]
                dist = float(torch.sqrt(dx * dx + dy * dy).item())

                if dist > 0:  # Skip identical positions
                    # Activity correlation across time and batch
                    act_i = s[:, :, i].reshape(-1)  # [B*T]
                    act_j = s[:, :, j].reshape(-1)  # [B*T]

                    # Compute correlation if there's variation
                    if act_i.var() > 1e-12 and act_j.var() > 1e-12:
                        corr = torch.corrcoef(torch.stack([act_i, act_j]))[0, 1]
                        if not torch.isnan(corr):
                            distances.append(dist)
                            correlations.append(abs(corr.item()))

        # If we have enough distance-correlation pairs, fit power law
        if len(distances) >= 3:
            distances = torch.tensor(distances, dtype=torch.float32)
            correlations = torch.tensor(correlations, dtype=torch.float32)

            # Fit power law: correlation ~ distance^(-beta)
            # Take log: log(corr) = -beta * log(dist) + const
            valid_mask = (distances > 0) & (correlations > 1e-12)
            if valid_mask.sum() >= 3:
                log_dist = torch.log(distances[valid_mask])
                log_corr = torch.log(correlations[valid_mask])

                A = torch.stack([log_dist, torch.ones_like(log_dist)], dim=1)
                try:
                    solution = torch.linalg.lstsq(A, log_corr.unsqueeze(1))
                    beta = float(-solution.solution[0].item())  # Negative slope -> positive beta

                    return beta, (beta - target_beta) ** 2
                except Exception:
                    pass

    # Method 3: Simple variance-based spatial measure (last resort)
    if D >= 1:
        s = s_hist[:, 1:, :]  # [B, T, D]

        # Compute spatial variance across neurons at each time point
        try:
            spatial_var = s.var(dim=-1, unbiased=False)  # [B, T] - variance across neurons
            mean_spatial_var = spatial_var.mean().item()

            # Use spatial variance as a proxy - critical systems should have moderate variance
            # Map variance to a pseudo-beta value
            if mean_spatial_var > 1e-12:
                # Heuristic mapping: higher variance -> lower beta (more critical)
                pseudo_beta = 2.0 + torch.log10(torch.tensor(mean_spatial_var + 1e-12)).item()
                pseudo_beta = max(0.1, min(4.0, pseudo_beta))  # Clamp to reasonable range

                return pseudo_beta, (pseudo_beta - target_beta) ** 2
        except Exception:
            pass

    return float("nan"), float("nan")


def susceptibility_variance(model: SOENModelCore, s_histories: list[torch.Tensor], *, layers: list[int] | None = None) -> float:
    # Compute variance of total network signal s(t) = sum_i s_i(t) summed over all neurons and layers.
    try:
        lids = _layer_ids(model)
        targets = layers or lids
        total_activity = None  # [B, T]
        for lid in targets:
            s = s_histories[lids.index(lid)][:, 1:, :]  # [B, T, D]
            layer_sum = s.sum(dim=-1)  # [B, T]
            total_activity = layer_sum if total_activity is None else (total_activity + layer_sum)
        if total_activity is None:
            return float("nan")
        # Variance over time and batch (population variance)
        chi = total_activity.var(unbiased=False).item()
        return float(chi)
    except Exception:
        return float("nan")


def criticality_avalanche_size_cost(s_histories: list[torch.Tensor]) -> float:
    """Avalanche size distribution cost - critical systems show power-law avalanche sizes.
    Measures variance in activity bursts as a proxy for scale-free avalanches.
    """
    try:
        # Compute activity bursts across all layers
        total_activity = torch.zeros_like(s_histories[0][:, 1:, 0])  # [B, T]
        for s in s_histories:
            total_activity += s[:, 1:, :].sum(dim=-1)  # Sum over neurons

        # Find avalanche events (above-threshold activity)
        activity_1d = total_activity.mean(dim=0)  # Average over batch: [T]
        threshold = activity_1d.mean() + 0.5 * activity_1d.std()

        # Identify avalanche sizes (consecutive above-threshold periods)
        above_thresh = (activity_1d > threshold).float()
        avalanche_sizes = []
        current_size = 0

        for active in above_thresh:
            if active > 0.5:
                current_size += 1
            else:
                if current_size > 0:
                    avalanche_sizes.append(current_size)
                current_size = 0

        if current_size > 0:
            avalanche_sizes.append(current_size)

        if len(avalanche_sizes) < 5:
            return 10.0  # Heavy penalty for no avalanches

        # Critical systems should have high variance in avalanche sizes (power-law)
        sizes = torch.tensor(avalanche_sizes, dtype=torch.float32)
        size_var = sizes.var().item()
        size_mean = sizes.mean().item()

        # Coefficient of variation - critical systems have high CV
        cv = size_var / (size_mean + 1e-6)

        # Cost: encourage high coefficient of variation (scale-free avalanches)
        target_cv = 2.0  # Typical for critical systems
        return abs(cv - target_cv)

    except Exception:
        return 10.0


def criticality_autocorr_cost(s_histories: list[torch.Tensor], dt_s: float) -> float:
    """Autocorrelation decay cost - critical systems show slow (power-law) decay."""
    try:
        # Compute total activity time series
        total_activity = torch.zeros(s_histories[0].shape[1] - 1)  # [T]
        for s in s_histories:
            total_activity += s[0, 1:, :].sum(dim=-1)  # First batch, sum over neurons

        # Compute autocorrelation using FFT (robust and available on all PyTorch versions)
        activity_centered = (total_activity - total_activity.mean()).to(dtype=torch.float32)
        L = activity_centered.numel()
        if L < 4:
            return 10.0
        nfft = 1
        while nfft < (2 * L - 1):
            nfft <<= 1
        X = torch.fft.rfft(activity_centered, n=nfft)
        S = X.conj() * X
        autocorr_full = torch.fft.irfft(S, n=nfft).real
        autocorr = autocorr_full[:L]
        if autocorr[0].abs() < 1e-12:
            return 10.0
        autocorr = autocorr / autocorr[0]

        # Fit power-law decay to autocorr: C(τ) ~ τ^(-α)
        max_lag = min(len(autocorr) // 4, 50)  # Don't use too long lags
        if max_lag < 10:
            return 10.0

        lags = torch.arange(1, max_lag + 1, dtype=torch.float32)
        autocorr_vals = autocorr[1 : max_lag + 1]

        # Remove negative/zero values for log
        valid_mask = autocorr_vals > 1e-6
        if valid_mask.sum() < 5:
            return 10.0

        log_lags = torch.log(lags[valid_mask])
        log_autocorr = torch.log(autocorr_vals[valid_mask])

        # Linear regression in log-log space
        A = torch.stack([log_lags, torch.ones_like(log_lags)], dim=1)
        try:
            coeffs = torch.linalg.lstsq(A, log_autocorr.unsqueeze(1)).solution.flatten()
            alpha = -coeffs[0].item()  # Power-law exponent
        except Exception:
            return 10.0

        # Critical systems typically have α ∈ [0.5, 1.5]
        target_alpha = 1.0
        return abs(alpha - target_alpha)

    except Exception:
        return 10.0


def _layer_dim_map(model: SOENModelCore, s_histories: list[torch.Tensor]) -> dict[int, int]:
    lids = _layer_ids(model)
    return {lid: int(s_histories[lids.index(lid)].shape[-1]) for lid in lids}


def _get_layer_attr_vector(layer, name: str, dim: int, device: torch.device) -> torch.Tensor:
    try:
        v = getattr(layer, name, None)
        if v is None:
            return torch.zeros(dim, device=device)
        # Parameters may be properties (e.g., exp(log_param)) → ensure tensor 1D
        vt = torch.as_tensor(v, device=device, dtype=torch.float32)
        if vt.dim() == 0:
            return vt.view(1).expand(dim).clone()
        if vt.dim() == 1 and vt.numel() == dim:
            return vt
        # Best-effort broadcast
        return vt.reshape(-1)[0:1].expand(dim).clone()
    except Exception:
        return torch.zeros(dim, device=device)


def _compute_source_derivatives(layer, phi_bt_d: torch.Tensor, eff_bias_bt_d: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (dg_dphi_mean, dg_dbias_mean) per neuron averaged over batch/time using autograd.

    Shapes:
      - phi_bt_d: [B,T,D]
      - eff_bias_bt_d: [B,T,D] or None
      Returns two tensors of shape [D].
    """
    _B, _T, D = phi_bt_d.shape
    device = phi_bt_d.device
    phi = phi_bt_d.detach().requires_grad_(True)
    if eff_bias_bt_d is not None:
        eff_bias = eff_bias_bt_d.detach().requires_grad_(True)
        g = layer.source_function.g(phi, squid_current=eff_bias)
        gmean = g.mean()
        dphi, dbias = torch.autograd.grad(gmean, (phi, eff_bias), retain_graph=False, create_graph=False)
        # Average over batch/time, keep per-neuron (already mean via gmean)
        dg_dphi = dphi.mean(dim=(0, 1))
        dg_dbias = dbias.mean(dim=(0, 1))
    else:
        g = layer.source_function.g(phi)
        gmean = g.mean()
        dphi = torch.autograd.grad(gmean, phi, retain_graph=False, create_graph=False)[0]
        dg_dphi = dphi.mean(dim=(0, 1))
        dg_dbias = torch.zeros(D, device=device)
    return dg_dphi, dg_dbias


def jacobian_spectral_radius(
    model: SOENModelCore,
    s_histories: list[torch.Tensor],
    *,
    target_layers: list[int] | None = None,
) -> tuple[float, dict]:
    """Estimate spectral radius of ds_t/ds_{t-1} directly from states via least squares.

    Builds X = concat_l s^{(l)}_{t-1} and Y = concat_l s^{(l)}_{t} across all target layers,
    stacks over batch/time, solves Y ≈ X W in LS sense, and returns ρ(W).

    This uses only state histories, no source-function details, and averages over
    all samples/time steps implicitly through the regression.
    """
    lids_all = _layer_ids(model)
    dims_all = _layer_dim_map(model, s_histories)
    # Determine target lids
    lids = target_layers or lids_all
    lids = [lid for lid in lids if lid in dims_all]
    if not lids:
        return float("nan"), {"reason": "no_target_layers"}

    # Build X, Y by concatenating per-layer states at t-1 and t
    X_parts = []
    Y_parts = []
    for lid in lids:
        idx = lids_all.index(lid)
        s = s_histories[idx]  # [B, T+1, D]
        if s.dim() != 3 or s.shape[1] < 2:
            continue
        X_parts.append(s[:, :-1, :])  # [B,T,D]
        Y_parts.append(s[:, 1:, :])  # [B,T,D]
    if not X_parts:
        return float("nan"), {"reason": "no_valid_histories"}
    # Concatenate along feature dim → [B,T,sumD]
    X_bt_D = torch.cat(X_parts, dim=-1)
    Y_bt_D = torch.cat(Y_parts, dim=-1)
    # Flatten B,T → N
    X = X_bt_D.reshape(-1, X_bt_D.shape[-1])
    Y = Y_bt_D.reshape(-1, Y_bt_D.shape[-1])
    # Guard: need at least a few rows
    if X.shape[0] < X.shape[1]:
        # Not enough samples to estimate full map
        return float("nan"), {"reason": "underdetermined", "rows": int(X.shape[0]), "cols": int(X.shape[1])}
    # Center data (remove intercept) so A approximates local Jacobian ∂s_t/∂s_{t-1}
    X_mean = X.mean(dim=0, keepdim=True)
    Y_mean = Y.mean(dim=0, keepdim=True)
    Xc = X - X_mean
    Yc = Y - Y_mean
    # Solve LS: Xc A = Yc
    try:
        sol = torch.linalg.lstsq(Xc, Yc)
        A = sol.solution  # [D, D]
        evals = torch.linalg.eigvals(A)
        rho = float(evals.abs().max().item())
        return rho, {"rows": int(X.shape[0]), "dim": int(X.shape[1])}
    except Exception as e:
        return float("nan"), {"error": str(e)}


def _to_1d_param(v: Any, dim: int, device: torch.device) -> torch.Tensor:
    try:
        t = torch.as_tensor(v, dtype=torch.float32, device=device)
        if t.dim() == 0:
            return t.view(1).expand(dim).clone()
        if t.dim() == 1 and t.numel() == dim:
            return t
        return t.reshape(-1)[0:1].expand(dim).clone()
    except Exception:
        return torch.zeros(dim, device=device)


def _dg_dphi_autograd(layer: Any, phi_bt_d: torch.Tensor, s_bt_d: torch.Tensor | None) -> torch.Tensor:
    """Compute ∂g/∂phi using autograd for a layer's source function.

    Args:
        layer: Layer module with `source_function` and optional `bias_current`.
        phi_bt_d: Tensor [B, T, D] of phi values.
        s_bt_d:   Tensor [B, T, D] of states or None.

    Returns:
        Tensor [B, T, D] of dg/dphi.

    """
    B, T, D = phi_bt_d.shape
    device = phi_bt_d.device
    phi = phi_bt_d.detach().clone().requires_grad_(True)
    try:
        bias = getattr(layer, "bias_current", None)
        if bias is None:
            bias_bc = None
        else:
            bias_1d = torch.as_tensor(bias, dtype=phi.dtype, device=device)
            if bias_1d.dim() == 0:
                bias_bc = bias_1d.view(1, 1, 1).expand(B, T, D)
            elif bias_1d.dim() == 1:
                bias_bc = bias_1d.view(1, 1, D).expand(B, T, D)
            else:
                bias_bc = bias_1d.reshape(1, 1, D).expand(B, T, D)
        if getattr(getattr(layer, "source_function", None), "uses_squid_current", False) and (s_bt_d is not None):
            if bias_bc is None:
                bias_bc = torch.full_like(phi, 1.7)
            squid_current = bias_bc - s_bt_d.detach()
        else:
            squid_current = bias_bc
        g_val = layer.source_function.g(phi, squid_current=squid_current)
        torch.autograd.backward(g_val, grad_tensors=torch.ones_like(g_val), retain_graph=False)
        dg_dphi = phi.grad
        if dg_dphi is None:
            msg = "autograd returned None for dg/dphi"
            raise RuntimeError(msg)
        return dg_dphi.detach()
    except Exception:
        eps = 1e-4
        phi_p = (phi_bt_d + eps).detach()
        phi_m = (phi_bt_d - eps).detach()
        if getattr(getattr(layer, "source_function", None), "uses_squid_current", False) and (s_bt_d is not None):
            squid_current = getattr(layer, "bias_current", 1.7)
            squid_current = torch.as_tensor(squid_current, dtype=phi_bt_d.dtype, device=device)
            if squid_current.dim() == 0:
                squid_current = squid_current.view(1, 1, 1).expand_as(phi_bt_d)
            elif squid_current.dim() == 1:
                squid_current = squid_current.view(1, 1, -1).expand_as(phi_bt_d)
        else:
            squid_current = getattr(layer, "bias_current", None)
            if squid_current is not None:
                squid_current = torch.as_tensor(squid_current, dtype=phi_bt_d.dtype, device=device)
                if squid_current.dim() == 0:
                    squid_current = squid_current.view(1, 1, 1).expand_as(phi_bt_d)
                elif squid_current.dim() == 1:
                    squid_current = squid_current.view(1, 1, -1).expand_as(phi_bt_d)
        try:
            gp = layer.source_function.g(phi_p, squid_current=squid_current)
            gm = layer.source_function.g(phi_m, squid_current=squid_current)
            return ((gp - gm) / (2.0 * eps)).detach()
        except Exception:
            return torch.zeros_like(phi_bt_d)


def lyapunov_largest_exponent(
    model: SOENModelCore,
    s_histories: list[torch.Tensor],
    dt_s: float,
    *,
    target_layers: list[int] | None = None,
    seed: int = 123,
    burn_in_frac: float = 0.1,
) -> tuple[float, float]:
    """Estimate the largest Lyapunov exponent using a Benettin-style update.

    Linearization uses Euler map A_t = I + dt * J(t) with
    J_l(t) ≈ -gamma_minus I + gamma_plus diag(g'(phi_l(t))) W_in(l),
    where W_in(l) aggregates incoming (and internal) connectivity.

    Returns: (lambda_per_step, lambda_per_second)
    """
    lids_all = _layer_ids(model)
    dims_all = _layer_dim_map(model, s_histories)
    if not s_histories or not lids_all:
        return float("nan"), float("nan")
    device = s_histories[0].device

    # Dynamic layers: require gamma_plus and gamma_minus
    dyn_lids: list[int] = []
    for lid in lids_all:
        idx = lids_all.index(lid)
        layer = model.layers[idx]
        if hasattr(layer, "gamma_plus") and hasattr(layer, "gamma_minus"):
            dyn_lids.append(lid)

    if target_layers is not None:
        dyn_lids = [lid for lid in dyn_lids if lid in set(target_layers)]
    if not dyn_lids:
        return float("nan"), float("nan")

    T = s_histories[0].shape[1] - 1
    if T <= 1:
        return float("nan"), float("nan")
    try:
        dt_units = float(model.dt) if hasattr(model, "dt") else float(getattr(model.sim_config, "dt", 1.0))
    except Exception:
        dt_units = dt_s if (dt_s == dt_s) else 1.0

    # Phi inputs per layer for batch 0
    phi_in = compute_phi_in_all_layers_adjusted(model, s_histories, subtract_phi_offset=False)
    phi_curr_by_lid: dict[int, torch.Tensor] = {}
    s_curr_by_lid: dict[int, torch.Tensor] = {}
    for lid in lids_all:
        if lid not in dims_all:
            continue
        D = dims_all[lid]
        s_bt_d = s_histories[lids_all.index(lid)][0, 1:, :].detach()  # [T, D]
        s_curr_by_lid[lid] = s_bt_d
        phi_bt_d = phi_in.get(lid)
        if phi_bt_d is None:
            phi_bt_d = torch.zeros(1, T, D, device=device)
        # Add phi_offset if present
        off = _phi_offset_vec(model, lid)
        if off is not None:
            phi_bt_d = phi_bt_d + off.view(1, 1, -1)
        phi_curr_by_lid[lid] = phi_bt_d[0]  # [T, D]

    # Derivatives of source nonlinearity
    dg_by_lid: dict[int, torch.Tensor] = {}
    for lid in dyn_lids:
        idx = lids_all.index(lid)
        layer = model.layers[idx]
        phi_td = phi_curr_by_lid.get(lid)
        s_td = s_curr_by_lid.get(lid)
        if phi_td is None or s_td is None:
            dg_by_lid[lid] = torch.zeros(T, dims_all[lid], device=device)
            continue
        dg = _dg_dphi_autograd(layer, phi_td.view(1, T, -1), s_td.view(1, T, -1))[0]
        dg_by_lid[lid] = dg  # [T, D]

    # Incoming connectivity per layer
    incoming: dict[int, list[tuple[int, torch.Tensor]]] = {lid: [] for lid in dyn_lids}
    for name, J in model.connections.items():
        if not name.startswith("J_"):
            continue
        f, t = _parse_conn_key(name)
        if t in incoming:
            incoming[t].append((f, J.detach()))

    # Internal connectivity
    internal_J: dict[int, torch.Tensor] = {}
    for lid in dyn_lids:
        layer = model.layers[lids_all.index(lid)]
        J_int = getattr(layer, "internal_J", None)
        if J_int is not None:
            internal_J[lid] = J_int.detach()

    # Gamma parameters per neuron
    gamma_plus_by_lid: dict[int, torch.Tensor] = {}
    gamma_minus_by_lid: dict[int, torch.Tensor] = {}
    for lid in dyn_lids:
        layer = model.layers[lids_all.index(lid)]
        D = dims_all[lid]
        gp = _to_1d_param(getattr(layer, "gamma_plus", 0.0), D, device)
        gm = _to_1d_param(getattr(layer, "gamma_minus", 0.0), D, device)
        gamma_plus_by_lid[lid] = gp
        gamma_minus_by_lid[lid] = gm

    # Initialize random perturbation and normalize
    torch.manual_seed(int(seed) if seed is not None else 123)
    delta_by_lid: dict[int, torch.Tensor] = {lid: torch.randn(dims_all[lid], device=device) for lid in dyn_lids}

    def _global_norm(d: dict[int, torch.Tensor]) -> torch.Tensor:
        return torch.sqrt(sum((v * v).sum() for v in d.values()) + 1e-30)

    nrm = _global_norm(delta_by_lid)
    for lid in dyn_lids:
        delta_by_lid[lid] = delta_by_lid[lid] / nrm

    burn_in = max(2, int(burn_in_frac * T))
    log_acc = 0.0
    n_steps_acc = 0

    for t in range(T):
        next_by_lid: dict[int, torch.Tensor] = {}
        for lid in dyn_lids:
            D = dims_all[lid]
            delta_cur = delta_by_lid[lid]
            gp = gamma_plus_by_lid[lid]
            gm = gamma_minus_by_lid[lid]
            dg = dg_by_lid[lid][t]
            # Aggregate incoming perturbations
            agg = torch.zeros(D, device=device)
            for f, J in incoming.get(lid, []):
                if f in delta_by_lid:
                    agg = agg + torch.matmul(J, delta_by_lid[f])
            if lid in internal_J:
                agg = agg + torch.matmul(internal_J[lid], delta_cur)
            # Euler linearization step
            delta_new = delta_cur + dt_units * (-gm * delta_cur + gp * (dg * agg))
            next_by_lid[lid] = delta_new

        nrm_new = _global_norm(next_by_lid)
        if nrm_new.item() == 0.0:
            for lid in dyn_lids:
                next_by_lid[lid] = torch.randn_like(next_by_lid[lid]) * 1e-6
            nrm_new = _global_norm(next_by_lid)

        growth = (nrm_new / _global_norm(delta_by_lid)).clamp(min=1e-12)
        if t >= burn_in:
            log_acc += float(torch.log(growth).item())
            n_steps_acc += 1
        # Normalize
        for lid in dyn_lids:
            next_by_lid[lid] = next_by_lid[lid] / nrm_new
        delta_by_lid = next_by_lid

    if n_steps_acc == 0:
        return float("nan"), float("nan")
    lam_per_step = log_acc / n_steps_acc
    lam_per_sec = lam_per_step / float(dt_s) if (dt_s == dt_s) and (dt_s > 0) else float("nan")
    return float(lam_per_step), float(lam_per_sec)
