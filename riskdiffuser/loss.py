from typing import Any, Callable, Dict, List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from riskdiffuser.utils.normalizer import StateNormalizer


def extract_ego_future_from_decoder_output(
    decoder_output: Dict[str, torch.Tensor],
    model_type: str,
    norm: StateNormalizer,
):
    if model_type != "x_start":
        raise NotImplementedError("risk order loss currently only supports diffusion_model_type='x_start'")

    pred_future = norm.inverse(decoder_output["score"][:, :, 1:, :])
    return pred_future[:, 0]


def compute_progress_from_ego_future(ego_future: torch.Tensor):
    delta_xy = ego_future[:, 1:, :2] - ego_future[:, :-1, :2]
    step_dist = torch.norm(delta_xy, dim=-1)
    return step_dist.sum(dim=-1, keepdim=True)


def compute_ego_jerk_loss_from_future(
    ego_future: torch.Tensor,
    dt: float = 0.1,
):
    """
    ego_future: [B, T, 4], containing x, y, cos, sin
    Computes jerk smoothness on xy trajectory only.
    """
    if ego_future.shape[1] < 4:
        return ego_future.new_tensor(0.0), {
            "ego_jerk_abs_mean": ego_future.new_tensor(0.0),
            "ego_jerk_l2_mean": ego_future.new_tensor(0.0),
        }

    velocity = (ego_future[:, 1:, :2] - ego_future[:, :-1, :2]) / dt
    acceleration = (velocity[:, 1:, :] - velocity[:, :-1, :]) / dt
    jerk = (acceleration[:, 1:, :] - acceleration[:, :-1, :]) / dt

    jerk_norm = torch.norm(jerk, dim=-1)
    jerk_loss = jerk_norm.pow(2).mean()

    metrics = {
        "ego_jerk_abs_mean": jerk_norm.mean().detach(),
        "ego_jerk_l2_mean": jerk_loss.detach(),
    }
    return jerk_loss, metrics

def compute_risk_order_loss(
    decoder_output_ref: Dict[str, torch.Tensor],
    decoder_output_rand: Dict[str, torch.Tensor],
    rho_ref: torch.Tensor,
    rho_rand: torch.Tensor,
    model_type: str,
    norm: StateNormalizer,
):
    ego_future_ref = extract_ego_future_from_decoder_output(decoder_output_ref, model_type, norm)
    ego_future_rand = extract_ego_future_from_decoder_output(decoder_output_rand, model_type, norm)

    progress_ref = compute_progress_from_ego_future(ego_future_ref)
    progress_rand = compute_progress_from_ego_future(ego_future_rand)

    delta_rho = rho_rand - rho_ref
    delta_progress = progress_ref - progress_rand

    signed_progress_gap = delta_rho * delta_progress
    progress_rank = F.relu(-signed_progress_gap)

    progress_rank_loss = progress_rank.mean()
    risk_order_loss = progress_rank_loss

    metrics = {
        "progress_ref_mean": progress_ref.mean().detach(),
        "progress_rand_mean": progress_rand.mean().detach(),
        "progress_gap_mean": delta_progress.abs().mean().detach(),
        "signed_progress_gap_mean": signed_progress_gap.mean().detach(),
        "progress_rank_loss": progress_rank_loss.detach(),
        "rho_gap_mean": delta_rho.abs().mean().detach(),
        "rho_pair_valid_ratio": torch.ones((), device=delta_rho.device),
    }

    return risk_order_loss, metrics


def diffusion_loss_func(
    model: nn.Module,
    inputs: Dict[str, torch.Tensor],
    marginal_prob: Callable[[torch.Tensor], torch.Tensor],

    futures: Tuple[torch.Tensor, torch.Tensor],
    
    norm: StateNormalizer,
    loss: Dict[str, Any],

    model_type: str,
    eps: float = 1e-3,
):   
    ego_future, neighbors_future, neighbor_future_mask = futures
    neighbors_future_valid = ~neighbor_future_mask # [B, P, V]

    B, Pn, T, _ = neighbors_future.shape
    ego_current, neighbors_current = inputs["ego_current_state"][:, :4], inputs["neighbor_agents_past"][:, :Pn, -1, :4]
    neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
    neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), neighbor_future_mask), dim=-1)

    gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future[..., :]], dim=1) # [B, P = 1 + 1 + neighbor, T, 4]
    current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1) # [B, P, 4]

    P = gt_future.shape[1]
    t = torch.rand(B, device=gt_future.device) * (1 - eps) + eps # [B,]
    z = torch.randn_like(gt_future, device=gt_future.device) # [B, P, T, 4]
    
    all_gt = torch.cat([current_states[:, :, None, :], norm(gt_future)], dim=2)
    all_gt[:, 1:][neighbor_mask] = 0.0

    mean, std = marginal_prob(all_gt[..., 1:, :], t)
    std = std.view(-1, *([1] * (len(all_gt[..., 1:, :].shape)-1)))

    xT = mean + std * z
    xT = torch.cat([all_gt[:, :, :1, :], xT], dim=2)
    
    merged_inputs = {
        **inputs,
        "sampled_trajectories": xT,
        "diffusion_time": t,
    }

    _, decoder_output = model(merged_inputs) # [B, P, 1 + T, 4]
    score = decoder_output["score"][:, :, 1:, :] # [B, P, T, 4]

    if model_type == "score":
        dpm_loss = torch.sum((score * std + z)**2, dim=-1)
    elif model_type == "x_start":
        dpm_loss = torch.sum((score - all_gt[:, :, 1:, :])**2, dim=-1)
    
    masked_prediction_loss = dpm_loss[:, 1:, :][neighbors_future_valid]

    if masked_prediction_loss.numel() > 0:
        loss["neighbor_prediction_loss"] = masked_prediction_loss.mean()
    else:
        loss["neighbor_prediction_loss"] = torch.tensor(0.0, device=masked_prediction_loss.device)

    loss["ego_planning_loss"] = dpm_loss[:, 0, :].mean()

    pred_future = norm.inverse(decoder_output["score"][:, :, 1:, :])
    pred_ego_future = pred_future[:, 0]
    jerk_loss, jerk_metrics = compute_ego_jerk_loss_from_future(pred_ego_future)
    loss["jerk_loss"] = jerk_loss
    loss.update(jerk_metrics)

    assert not torch.isnan(dpm_loss).sum(), f"loss cannot be nan, z={z}"

    return loss, decoder_output, t
