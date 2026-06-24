from tqdm import tqdm
import torch
from torch import nn

from riskdiffuser.utils.data_augmentation import StatePerturbation
from riskdiffuser.utils.train_utils import get_epoch_mean_loss
from riskdiffuser.utils import ddp
from riskdiffuser.loss import diffusion_loss_func, compute_risk_order_loss


def sample_separated_rho(rho_ref, rho_min, rho_max, min_gap):
    rho_low_bound = torch.full_like(rho_ref, rho_min)
    rho_high_bound = torch.full_like(rho_ref, rho_max)
    gap = torch.full_like(rho_ref, min_gap)

    left_max = torch.clamp(rho_ref - gap, min=rho_min, max=rho_max)
    right_min = torch.clamp(rho_ref + gap, min=rho_min, max=rho_max)

    left_width = (left_max - rho_low_bound).clamp(min=0.0)
    right_width = (rho_high_bound - right_min).clamp(min=0.0)

    choose_right = torch.rand_like(rho_ref) > 0.5
    choose_right = torch.where(left_width <= 1e-6, torch.ones_like(choose_right, dtype=torch.bool), choose_right)
    choose_right = torch.where(right_width <= 1e-6, torch.zeros_like(choose_right, dtype=torch.bool), choose_right)

    rand_unit = torch.rand_like(rho_ref)
    rho_left = rho_low_bound + rand_unit * left_width
    rho_right = right_min + rand_unit * right_width
    rho_rand = torch.where(choose_right, rho_right, rho_left)

    return rho_rand.clamp(rho_min, rho_max)


def train_epoch(data_loader, model, optimizer, args, ema, aug: StatePerturbation=None, risk_model=None):
    epoch_loss = []

    model.train()
    if risk_model is not None:
        risk_model.eval()

    if args.ddp:
        torch.cuda.synchronize()

    with tqdm(data_loader, desc="Training", unit="batch") as data_epoch:
        for batch in data_epoch:
            '''
            data structure in batch: Tuple(Tensor) 

            ego_current_state,
            ego_future_gt,

            neighbor_agents_past,
            neighbors_future_gt,

            lanes,
            lanes_speed_limit,
            lanes_has_speed_limit,

            route_lanes,
            route_lanes_speed_limit,
            route_lanes_has_speed_limit,

            static_objects,

            '''

            # prepare data
            inputs = {
                'ego_current_state': batch[0].to(args.device),

                'neighbor_agents_past': batch[2].to(args.device),

                'lanes': batch[4].to(args.device),
                'lanes_speed_limit': batch[5].to(args.device),
                'lanes_has_speed_limit': batch[6].to(args.device),

                'route_lanes': batch[7].to(args.device),
                'route_lanes_speed_limit': batch[8].to(args.device),
                'route_lanes_has_speed_limit': batch[9].to(args.device),
                'static_objects': batch[10].to(args.device),
            }

            ego_future = batch[1].to(args.device)
            neighbors_future = batch[3].to(args.device)
            # Normalize to ego-centric
            if aug is not None:
                inputs, ego_future, neighbors_future = aug(inputs, ego_future, neighbors_future)

            # heading to cos sin
            ego_future = torch.cat(
            [
                ego_future[..., :2],
                torch.stack(
                    [ego_future[..., 2].cos(), ego_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )

            mask = torch.sum(torch.ne(neighbors_future[..., :3], 0), dim=-1) == 0
            neighbors_future = torch.cat(
            [
                neighbors_future[..., :2],
                torch.stack(
                    [neighbors_future[..., 2].cos(), neighbors_future[..., 2].sin()], dim=-1
                ),
            ],
            dim=-1,
            )
            neighbors_future[mask] = 0.

            normalized_inputs = args.observation_normalizer(inputs)

            if risk_model is not None:
                with torch.no_grad():
                    risk_outputs = risk_model(normalized_inputs)
                    rho = risk_outputs["risk_score"].detach()
                    if args.shuffle_inferred_rho_within_batch:
                        rho = 1.0 - rho
            else:
                rho = batch[11].to(args.device)

            inputs = normalized_inputs
            inputs["risk_level"] = rho
                  
            # call the mdoel
            optimizer.zero_grad()
            loss = {}

            loss, _, _ = diffusion_loss_func(
                model,
                inputs,
                ddp.get_model(model, args.ddp).sde.marginal_prob,
                (ego_future, neighbors_future, mask),
                args.state_normalizer,
                loss,
                args.diffusion_model_type
            )

            total_loss = loss['neighbor_prediction_loss'] + args.alpha_planning_loss * loss['ego_planning_loss']

            if args.use_jerk_loss:
                total_loss = total_loss + args.alpha_jerk_loss * loss["jerk_loss"]
            else:
                loss["jerk_loss"] = torch.tensor(0.0, device=rho.device)
                loss["ego_jerk_abs_mean"] = torch.tensor(0.0, device=rho.device)
                loss["ego_jerk_l2_mean"] = torch.tensor(0.0, device=rho.device)

            if args.use_risk_order_loss:
                if args.diffusion_model_type != "x_start":
                    raise NotImplementedError("risk order loss currently requires diffusion_model_type='x_start'")

                ego_current = inputs["ego_current_state"][:, :4]
                neighbors_current = inputs["neighbor_agents_past"][:, :args.predicted_neighbor_num, -1, :4]
                neighbor_current_mask = torch.sum(torch.ne(neighbors_current[..., :4], 0), dim=-1) == 0
                neighbor_mask = torch.concat((neighbor_current_mask.unsqueeze(-1), mask), dim=-1)

                gt_future = torch.cat([ego_future[:, None, :, :], neighbors_future], dim=1)
                current_states = torch.cat([ego_current[:, None], neighbors_current], dim=1)
                sampled_gt = torch.cat([current_states[:, :, None, :], args.state_normalizer(gt_future)], dim=2)
                sampled_gt[:, 1:][neighbor_mask] = 0.0

                rho_rand = sample_separated_rho(
                    rho_ref=rho,
                    rho_min=args.random_rho_min,
                    rho_max=args.random_rho_max,
                    min_gap=args.risk_order_rho_threshold,
                )
                pair_t = torch.full((rho.shape[0],), 1e-3, device=rho.device)

                inputs_ref = {
                    **inputs,
                    "risk_level": rho,
                    "sampled_trajectories": sampled_gt,
                    "diffusion_time": pair_t,
                }
                inputs_rand = {
                    **inputs,
                    "risk_level": rho_rand,
                    "sampled_trajectories": sampled_gt,
                    "diffusion_time": pair_t,
                }

                _, decoder_output_ref = model(inputs_ref)
                _, decoder_output_rand = model(inputs_rand)

                risk_order_loss, risk_metrics = compute_risk_order_loss(
                    decoder_output_ref=decoder_output_ref,
                    decoder_output_rand=decoder_output_rand,
                    rho_ref=rho,
                    rho_rand=rho_rand,
                    model_type=args.diffusion_model_type,
                    norm=args.state_normalizer,
                )
                loss["risk_order_loss"] = risk_order_loss
                loss.update(risk_metrics)
                total_loss = total_loss + args.alpha_risk_order_loss * risk_order_loss
            else:
                loss["risk_order_loss"] = torch.tensor(0.0, device=rho.device)

            loss['loss'] = total_loss

            total_loss = loss['loss'].item()

            # loss backward
            loss['loss'].backward()

            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()

            if ema is not None:
                ema.update(model)

            if args.ddp:
                torch.cuda.synchronize()
            
            data_epoch.set_postfix(loss='{:.4f}'.format(total_loss))
            epoch_loss.append(loss)

    epoch_mean_loss = get_epoch_mean_loss(epoch_loss)

    if args.ddp:
        epoch_mean_loss = ddp.reduce_and_average_losses(epoch_mean_loss, torch.device(args.device))

    if ddp.get_rank() == 0:
        print(f"epoch train loss: {epoch_mean_loss['loss']:.4f}\n")
        
    return epoch_mean_loss, epoch_mean_loss['loss']
