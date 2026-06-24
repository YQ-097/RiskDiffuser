import torch
import torch.nn as nn
import torch.nn.functional as F


class RiskAssessmentHead(nn.Module):
    def __init__(self, hidden_dim, agent_num, static_num, lane_num):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.agent_num = agent_num
        self.static_num = static_num
        self.lane_num = lane_num

        in_dim = hidden_dim * 4

        self.fusion = nn.Sequential(
            nn.Linear(in_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.risk_head = nn.Linear(hidden_dim, 1)
        self.ttc_head = nn.Linear(hidden_dim, 1)
        self.dist_head = nn.Linear(hidden_dim, 1)

    @staticmethod
    def masked_mean_pool(x, mask):
        """
        x:    [B, N, D]
        mask: [B, N]  True means invalid
        """
        valid = (~mask).float().unsqueeze(-1)   # [B,N,1]
        x = x * valid
        denom = valid.sum(dim=1).clamp(min=1.0)
        return x.sum(dim=1) / denom

    def forward(self, encoding, encoding_mask):
        """
        encoding:      [B, token_num, D]
        encoding_mask: [B, token_num]  True means invalid
        """
        B, T, D = encoding.shape

        # token partition
        a0, a1 = 0, self.agent_num
        s0, s1 = a1, a1 + self.static_num
        l0, l1 = s1, s1 + self.lane_num

        agent_feat = self.masked_mean_pool(
            encoding[:, a0:a1], encoding_mask[:, a0:a1]
        )
        static_feat = self.masked_mean_pool(
            encoding[:, s0:s1], encoding_mask[:, s0:s1]
        )
        lane_feat = self.masked_mean_pool(
            encoding[:, l0:l1], encoding_mask[:, l0:l1]
        )
        global_feat = self.masked_mean_pool(
            encoding, encoding_mask
        )

        feat = torch.cat([global_feat, agent_feat, static_feat, lane_feat], dim=-1)
        feat = self.fusion(feat)

        risk_score = torch.sigmoid(self.risk_head(feat))
        ttc_proxy = torch.sigmoid(self.ttc_head(feat))
        dist_proxy = torch.sigmoid(self.dist_head(feat))

        return {
            "risk_score": risk_score,
            "ttc_proxy": ttc_proxy,
            "dist_proxy": dist_proxy,
            "risk_feat": feat,
        }
    
class SharedEncoderRiskNet(nn.Module):
    def __init__(self, encoder, config):
        super().__init__()
        self.encoder = encoder

        self.risk_head = RiskAssessmentHead(
            hidden_dim=config.hidden_dim,
            agent_num=config.agent_num,
            static_num=config.static_objects_num,
            lane_num=config.lane_num,
        )

    def forward(self, inputs):
        encoder_outputs = self.encoder(inputs)

        risk_outputs = self.risk_head(
            encoding=encoder_outputs["encoding"],
            encoding_mask=encoder_outputs["encoding_mask"],
        )

        return {
            **encoder_outputs,
            **risk_outputs,
        }
    
class RiskHeadLoss(nn.Module):
    def __init__(self, w_risk=1.0, w_ttc=0.2, w_dist=0.2, w_rank=0.1, rank_margin=0.03):
        super().__init__()
        self.w_risk = w_risk
        self.w_ttc = w_ttc
        self.w_dist = w_dist
        self.w_rank = w_rank
        self.rank_margin = rank_margin
        self.reg = nn.SmoothL1Loss()

    def pairwise_rank_loss(self, pred, target):
        pred = pred.squeeze(-1)
        target = target.squeeze(-1)

        diff_t = target.unsqueeze(1) - target.unsqueeze(0)
        diff_p = pred.unsqueeze(1) - pred.unsqueeze(0)

        valid = diff_t > 1e-4
        if valid.sum() == 0:
            return pred.new_tensor(0.0)

        return F.relu(self.rank_margin - diff_p[valid]).mean()

    def forward(self, outputs, teacher):

        loss_risk = self.reg(outputs["risk_score"], teacher["risk_target"])
        loss_ttc = self.reg(outputs["ttc_proxy"], teacher["r_ttc"])
        loss_dist = self.reg(outputs["dist_proxy"], teacher["r_dist"])
        loss_rank = self.pairwise_rank_loss(outputs["risk_score"], teacher["risk_target"])

        loss = (
            self.w_risk * loss_risk
            + self.w_ttc * loss_ttc
            + self.w_dist * loss_dist
            + self.w_rank * loss_rank
        )

        return {
            "loss": loss,
            "loss_risk": loss_risk,
            "loss_ttc": loss_ttc,
            "loss_dist": loss_dist,
            "loss_rank": loss_rank,
        }


def build_future_risk_teacher(
    ego_current_state,              # [B, D]
    ego_future_gt,                  # [B, T, 4]
    neighbor_agents_past,           # [B, Np, Tp, Dn]   # 这里只保留接口，不再用于 future mask
    neighbors_future_gt,            # [B, Nf, T, 4]
    route_lanes,                    # [B, P, V, Dl]
    route_lanes_speed_limit,        # [B, P]
    route_lanes_has_speed_limit,    # [B, P]
    static_objects,                 # [B, S, Ds]
    sigma_d=8.0,
    sigma_t=3.0,
    sigma_s=6.0,
    collision_dist=2.0,
    eps=1e-6,
):
    device = ego_current_state.device
    B = ego_current_state.shape[0]

    # =====================================================
    # 1) future distance risk
    # =====================================================
    ego_xy_future = ego_future_gt[..., :2]                        # [B,T,2]
    nbr_xy_future = neighbors_future_gt[..., :2]                  # [B,Nf,T,2]

    # 关键修复：
    # valid mask 必须和 neighbors_future_gt 对齐，而不是用 neighbor_agents_past
    nbr_valid_t = torch.abs(neighbors_future_gt[..., :2]).sum(dim=-1) > 0   # [B,Nf,T]
    nbr_valid = nbr_valid_t.any(dim=-1)                                      # [B,Nf]

    ego_expand = ego_xy_future[:, None, :, :]                     # [B,1,T,2]
    dist_future = torch.norm(ego_expand - nbr_xy_future, dim=-1) # [B,Nf,T]
    dist_future = dist_future.masked_fill(~nbr_valid[:, :, None], 1e6)

    dmin_future = dist_future.amin(dim=(1, 2))                    # [B]
    r_dist = torch.exp(-dmin_future / sigma_d)

    # =====================================================
    # 2) future collision indicator
    # =====================================================
    r_collision = (dmin_future < collision_dist).float()

    # =====================================================
    # 3) future TTC risk (approx from future finite differences)
    # =====================================================
    ego_vel_future = ego_xy_future[:, 1:, :] - ego_xy_future[:, :-1, :]          # [B,T-1,2]
    nbr_vel_future = nbr_xy_future[:, :, 1:, :] - nbr_xy_future[:, :, :-1, :]    # [B,Nf,T-1,2]

    ego_pos = ego_xy_future[:, :-1, :]                                            # [B,T-1,2]
    nbr_pos = nbr_xy_future[:, :, :-1, :]                                         # [B,Nf,T-1,2]

    rel_pos = nbr_pos - ego_pos[:, None, :, :]                                    # [B,Nf,T-1,2]
    rel_vel = nbr_vel_future - ego_vel_future[:, None, :, :]                      # [B,Nf,T-1,2]

    rel_dist_t = torch.norm(rel_pos, dim=-1)                                      # [B,Nf,T-1]

    # TTC 的 valid 也要和 T-1 对齐
    nbr_valid_ttc = nbr_valid_t[:, :, :-1]                                        # [B,Nf,T-1]
    rel_dist_t = rel_dist_t.masked_fill(~nbr_valid_ttc, 1e6)

    closing_speed = -(rel_pos * rel_vel).sum(dim=-1) / (rel_dist_t + eps)

    ttc_future = torch.where(
        (closing_speed > 0) & nbr_valid_ttc,
        rel_dist_t / (closing_speed + eps),
        torch.full_like(rel_dist_t, 1e6)
    )

    ttc_min_future = ttc_future.amin(dim=(1, 2)).clamp(max=10.0)                  # [B]
    r_ttc = torch.exp(-ttc_min_future / sigma_t)

    # =====================================================
    # 4) static object future risk
    # =====================================================
    if static_objects.shape[1] > 0:
        static_xy = static_objects[..., :2]                                       # [B,S,2]
        static_valid = torch.abs(static_objects).sum(dim=-1) > 0                  # [B,S]

        ego_expand_s = ego_xy_future[:, :, None, :]                               # [B,T,1,2]
        static_expand = static_xy[:, None, :, :]                                  # [B,1,S,2]

        dist_static = torch.norm(ego_expand_s - static_expand, dim=-1)            # [B,T,S]
        dist_static = dist_static.masked_fill(~static_valid[:, None, :], 1e6)

        dmin_static = dist_static.amin(dim=(1, 2))
        r_static = torch.exp(-dmin_static / sigma_s)
    else:
        r_static = torch.zeros(B, device=device)

        # =====================================================
    # 5) speed risk (current)
    # =====================================================
    if ego_current_state.shape[-1] >= 6:
        ego_speed = torch.norm(ego_current_state[:, 4:6], dim=-1)   # sqrt(vx^2 + vy^2)
    else:
        ego_speed = torch.zeros(B, device=device)

    valid_limit = (route_lanes_has_speed_limit > 0).reshape(B, -1)
    route_speed = route_lanes_speed_limit.reshape(B, -1)

    limit_sum = (route_speed * valid_limit.float()).sum(dim=1)
    limit_cnt = valid_limit.float().sum(dim=1)

    mean_limit = torch.where(
        limit_cnt > 0,
        limit_sum / (limit_cnt + eps),
        torch.zeros_like(limit_sum)
    )

    r_speed = torch.where(
        limit_cnt > 0,
        ((ego_speed - mean_limit) / (mean_limit + eps)).clamp(0.0, 1.0),
        torch.zeros_like(ego_speed)
    )

    # =====================================================
    # 6) route curvature risk
    # =====================================================
    route_pts = route_lanes[..., :2]                                              # [B,P,V,2]
    B_, P, V, _ = route_pts.shape
    route_pts = route_pts.reshape(B_, P * V, 2)

    vec1 = route_pts[:, 1:-1, :] - route_pts[:, :-2, :]
    vec2 = route_pts[:, 2:, :] - route_pts[:, 1:-1, :]

    n1 = torch.norm(vec1, dim=-1)
    n2 = torch.norm(vec2, dim=-1)
    valid_curve = (n1 > 1e-3) & (n2 > 1e-3)

    cosang = (vec1 * vec2).sum(dim=-1) / (n1 * n2 + eps)
    cosang = cosang.clamp(-1.0, 1.0)
    dtheta = torch.arccos(cosang)
    dtheta = torch.where(valid_curve, dtheta, torch.zeros_like(dtheta))
    curve_mean = dtheta.mean(dim=1)
    r_map = (curve_mean / 0.15).clamp(0.0, 1.0)

    # =====================================================
    # 7) total oracle risk teacher
    # =====================================================

    risk_target = (
        0.30 * r_dist +
        0.25 * r_ttc +
        0.20 * r_collision +
        0.15 * r_static +
        0.05 * r_speed +
        0.05 * r_map
    ).clamp(0.0, 1.0)

    assert r_dist.shape == (B,), f"r_dist shape error: {r_dist.shape}"
    assert r_ttc.shape == (B,), f"r_ttc shape error: {r_ttc.shape}"
    assert r_collision.shape == (B,), f"r_collision shape error: {r_collision.shape}"
    assert r_static.shape == (B,), f"r_static shape error: {r_static.shape}"
    assert r_speed.shape == (B,), f"r_speed shape error: {r_speed.shape}"
    assert r_map.shape == (B,), f"r_map shape error: {r_map.shape}"
    assert risk_target.shape == (B,), f"risk_target shape error: {risk_target.shape}"

    return {
        "risk_target": risk_target.unsqueeze(-1),
        "r_ttc": r_ttc.unsqueeze(-1),
        "r_dist": r_dist.unsqueeze(-1),
        "r_collision": r_collision.unsqueeze(-1),
        "r_static": r_static.unsqueeze(-1),
        "r_speed": r_speed.unsqueeze(-1),
        "r_map": r_map.unsqueeze(-1),
        "dmin_future": dmin_future.unsqueeze(-1),
        "ttc_min_future": ttc_min_future.unsqueeze(-1),
    }


##根据训练集的teacher risk分布，将RiskNet的输出风险分数校准到0-1的范围
class TeacherRiskCalibratorTorch:
    def __init__(self, device="cpu"):
        self.x = torch.tensor([
            0.011037,
            0.038780,
            0.191027,
            0.300727,
            0.405701,
            0.623933,
            0.777881,
            0.823313,
            0.885910,
        ], dtype=torch.float32, device=device)

        self.y = torch.tensor([
            0.00,
            0.01,
            0.05,
            0.25,
            0.50,
            0.75,
            0.95,
            0.99,
            1.00,
        ], dtype=torch.float32, device=device)

    def transform(self, values: torch.Tensor) -> torch.Tensor:
        values = values.float()

        # 先限制范围
        v = torch.clamp(values, self.x[0], self.x[-1])

        # 找到每个值所在区间
        idx = torch.searchsorted(self.x, v, right=True)
        idx = torch.clamp(idx, 1, len(self.x) - 1)

        x0 = self.x[idx - 1]
        x1 = self.x[idx]
        y0 = self.y[idx - 1]
        y1 = self.y[idx]

        t = (v - x0) / (x1 - x0 + 1e-12)
        out = y0 + t * (y1 - y0)
        return torch.clamp(out, 0.0, 1.0)

    def inverse_transform(self, q_values: torch.Tensor) -> torch.Tensor:
        q = q_values.float()
        q = torch.clamp(q, self.y[0], self.y[-1])

        idx = torch.searchsorted(self.y, q, right=True)
        idx = torch.clamp(idx, 1, len(self.y) - 1)

        y0 = self.y[idx - 1]
        y1 = self.y[idx]
        x0 = self.x[idx - 1]
        x1 = self.x[idx]

        t = (q - y0) / (y1 - y0 + 1e-12)
        out = x0 + t * (x1 - x0)
        return torch.clamp(out, self.x[0], self.x[-1])