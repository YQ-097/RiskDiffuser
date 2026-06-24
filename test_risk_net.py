import os
import io
import json
import argparse
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from mmengine import fileio
from timm.utils import ModelEma

from riskdiffuser.model.riskdiffuser import RiskDiffuser
from riskdiffuser.model.risknet import (
    SharedEncoderRiskNet,
    build_future_risk_teacher,
    TeacherRiskCalibratorTorch,
)
from riskdiffuser.utils.normalizer import ObservationNormalizer, StateNormalizer


def openjson(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def opendata(path):
    return np.load(path, allow_pickle=True)


def load_npz_string(x):
    if isinstance(x, np.ndarray):
        if x.shape == ():
            return str(x.item())
        return str(x.tolist())
    return str(x)


class RiskDiffuserTestData(Dataset):
    """
    测试专用：
    - 前 11 项与训练版保持一致
    - 额外返回 scenario_token / scenario_type
    - 不计算 risk_level，避免和训练逻辑混淆
    """

    def __init__(self, data_dir, data_list, past_neighbor_num, predicted_neighbor_num, future_len):
        self.data_dir = data_dir
        self.data_list = openjson(data_list)
        self._past_neighbor_num = past_neighbor_num
        self._predicted_neighbor_num = predicted_neighbor_num
        self._future_len = future_len

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = opendata(os.path.join(self.data_dir, self.data_list[idx]))

        ego_current_state = data["ego_current_state"][..., :10]
        ego_agent_future = data["ego_agent_future"]

        neighbor_agents_past = data["neighbor_agents_past"][:self._past_neighbor_num]
        neighbor_agents_future = data["neighbor_agents_future"][:self._predicted_neighbor_num]

        lanes = data["lanes"]
        lanes_speed_limit = data["lanes_speed_limit"]
        lanes_has_speed_limit = data["lanes_has_speed_limit"]

        route_lanes = data["route_lanes"]
        route_lanes_speed_limit = data["route_lanes_speed_limit"]
        route_lanes_has_speed_limit = data["route_lanes_has_speed_limit"]

        static_objects = data["static_objects"]

        scenario_token = load_npz_string(data["token"]) if "token" in data.files else os.path.splitext(self.data_list[idx])[0]
        scenario_type = load_npz_string(data["scenario_type"]) if "scenario_type" in data.files else "unknown"

        return (
            ego_current_state,             # 0
            ego_agent_future,             # 1
            neighbor_agents_past,         # 2
            neighbor_agents_future,       # 3
            lanes,                        # 4
            lanes_speed_limit,            # 5
            lanes_has_speed_limit,        # 6
            route_lanes,                  # 7
            route_lanes_speed_limit,      # 8
            route_lanes_has_speed_limit,  # 9
            static_objects,               # 10
            scenario_token,               # 11
            scenario_type,                # 12
        )


def boolean(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def get_args():
    parser = argparse.ArgumentParser(description="Test RiskNet on processed test set")

    parser.add_argument("--test_set", type=str, required=True)
    parser.add_argument("--test_set_list", type=str, required=True)
    parser.add_argument("--risk_ckpt", type=str, required=True)

    parser.add_argument("--output_dir", type=str, default="./risk_test_outputs")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--use_ema", type=boolean, default=True)
    parser.add_argument("--pin_mem", type=boolean, default=True)

    # dataset/model config
    parser.add_argument("--future_len", type=int, default=80)
    parser.add_argument("--time_len", type=int, default=21)

    parser.add_argument("--agent_state_dim", type=int, default=11)
    parser.add_argument("--agent_num", type=int, default=32)

    parser.add_argument("--static_objects_state_dim", type=int, default=10)
    parser.add_argument("--static_objects_num", type=int, default=5)

    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_state_dim", type=int, default=12)
    parser.add_argument("--lane_num", type=int, default=70)

    parser.add_argument("--route_len", type=int, default=20)
    parser.add_argument("--route_state_dim", type=int, default=12)
    parser.add_argument("--route_num", type=int, default=25)

    parser.add_argument("--predicted_neighbor_num", type=int, default=10)

    parser.add_argument("--encoder_depth", type=int, default=3)
    parser.add_argument("--decoder_depth", type=int, default=3)
    parser.add_argument("--num_heads", type=int, default=6)
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--diffusion_model_type", type=str, default="x_start")
    parser.add_argument("--encoder_drop_path_rate", type=float, default=0.1)
    parser.add_argument("--decoder_drop_path_rate", type=float, default=0.1)

    parser.add_argument("--normalization_file_path", type=str, default="normalization.json")

    args = parser.parse_args()
    args.state_normalizer = StateNormalizer.from_json(args)
    args.observation_normalizer = ObservationNormalizer.from_json(args)
    return args


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def tensor_to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def flatten_b1(x):
    x = tensor_to_numpy(x)
    return x.reshape(-1)


def pearson_corr(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if len(x) < 2:
        return np.nan
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(x, y):
    x = pd.Series(np.asarray(x, dtype=np.float64))
    y = pd.Series(np.asarray(y, dtype=np.float64))
    if len(x) < 2:
        return np.nan
    xr = x.rank(method="average").to_numpy()
    yr = y.rank(method="average").to_numpy()
    return pearson_corr(xr, yr)


def mse(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((x - y) ** 2))


def mae(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean(np.abs(x - y)))


def rmse(x, y):
    return float(np.sqrt(mse(x, y)))


def _resolve_ckpt_path(path: str) -> str:
    if os.path.isdir(path):
        for name in ["latest.pth", "model.pth"]:
            p = os.path.join(path, name)
            if os.path.exists(p):
                return p
        raise FileNotFoundError(f"No checkpoint found in directory: {path}")
    return path


def _load_ckpt(path: str, map_location="cpu"):
    real_path = _resolve_ckpt_path(path)
    ckpt_bytes = fileio.get(real_path)
    with io.BytesIO(ckpt_bytes) as f:
        ckpt = torch.load(f, map_location=map_location, weights_only=False)
    return ckpt, real_path


def load_risk_model(args, device):
    base_model = RiskDiffuser(args).to(device)
    risk_model = SharedEncoderRiskNet(base_model.encoder, args).to(device)

    ckpt, real_path = _load_ckpt(args.risk_ckpt, map_location=device)
    print(f"Loading RiskNet checkpoint from: {real_path}")

    state_dict = ckpt["model"] if "model" in ckpt else ckpt
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = risk_model.load_state_dict(state_dict, strict=False)

    print("RiskNet load done")
    print("Missing keys:", msg.missing_keys)
    print("Unexpected keys:", msg.unexpected_keys)

    model_for_eval = risk_model
    if args.use_ema and "ema_state_dict" in ckpt:
        print("Using EMA weights for evaluation")
        ema = ModelEma(risk_model, decay=0.999, device=device)
        ema_state = ckpt["ema_state_dict"]
        if isinstance(ema_state, dict) and "module" in ema_state:
            ema_state = ema_state["module"]
        ema_state = {k.replace("module.", ""): v for k, v in ema_state.items()}
        try:
            ema.ema.load_state_dict(ema_state, strict=False)
            model_for_eval = ema.ema
        except Exception as e:
            print(f"EMA load failed, fallback to raw model. Reason: {e}")

    model_for_eval.eval()
    return model_for_eval


def build_inputs_from_batch(batch, device):
    inputs = {
        "ego_current_state": batch[0].to(device),
        "neighbor_agents_past": batch[2].to(device),
        "lanes": batch[4].to(device),
        "lanes_speed_limit": batch[5].to(device),
        "lanes_has_speed_limit": batch[6].to(device),
        "route_lanes": batch[7].to(device),
        "route_lanes_speed_limit": batch[8].to(device),
        "route_lanes_has_speed_limit": batch[9].to(device),
        "static_objects": batch[10].to(device),
    }
    ego_future_gt = batch[1].to(device)
    neighbors_future_gt = batch[3].to(device)
    return inputs, ego_future_gt, neighbors_future_gt


def infer_batch_size(batch):
    for item in batch:
        if isinstance(item, torch.Tensor):
            return item.shape[0]
        if isinstance(item, (list, tuple)):
            return len(item)
        if isinstance(item, np.ndarray):
            return item.shape[0]
    raise RuntimeError("Cannot infer batch size from batch.")


def resolve_scenario_type(batch, batch_size):
    for idx in range(len(batch) - 1, max(-1, len(batch) - 6), -1):
        item = batch[idx]
        if isinstance(item, (list, tuple)) and len(item) == batch_size:
            if all(isinstance(v, str) for v in item):
                uniq = set(item)
                if len(uniq) < batch_size:
                    return list(item)
        if isinstance(item, np.ndarray) and item.shape[0] == batch_size:
            if item.dtype.kind in {"U", "S", "O"}:
                vals = item.tolist()
                uniq = set(vals)
                if len(uniq) < batch_size:
                    return vals
    return ["unknown"] * batch_size


def resolve_scenario_token(batch, batch_size, start_idx):
    if len(batch) > 11:
        item = batch[11]
        if isinstance(item, (list, tuple)) and len(item) == batch_size:
            return [str(v) for v in item]
        if isinstance(item, np.ndarray) and item.shape[0] == batch_size:
            return [str(v) for v in item.tolist()]
    return [f"sample_{start_idx + i:06d}" for i in range(batch_size)]


def summarize_global(df: pd.DataFrame, pred_col="pred_risk", tgt_col="teacher_risk_cdf") -> Dict[str, Any]:
    pred = df[pred_col].to_numpy()
    tgt = df[tgt_col].to_numpy()

    summary = {
        "num_samples": int(len(df)),
        "mae": mae(pred, tgt),
        "mse": mse(pred, tgt),
        "rmse": rmse(pred, tgt),
        "pearson": pearson_corr(pred, tgt),
        "spearman": spearman_corr(pred, tgt),
        "pred_mean": float(np.mean(pred)),
        "teacher_mean": float(np.mean(tgt)),
        "pred_std": float(np.std(pred)),
        "teacher_std": float(np.std(tgt)),
    }

    return summary


def summarize_by_scenario_type(df: pd.DataFrame, pred_col="pred_risk", tgt_col="teacher_risk_cdf") -> pd.DataFrame:
    count_col = "scenario_token" if "scenario_token" in df.columns else "sample_id"
    group_df = (
        df.groupby("scenario_type", dropna=False)
        .agg(
            count=(count_col, "count"),
            pred_risk_mean=(pred_col, "mean"),
            pred_risk_std=(pred_col, "std"),
            teacher_risk_mean=(tgt_col, "mean"),
            teacher_risk_std=(tgt_col, "std"),
        )
        .reset_index()
    )

    group_df["mae"] = (
        df.groupby("scenario_type", dropna=False)
        .apply(lambda g: float(np.mean(np.abs(g[pred_col].to_numpy() - g[tgt_col].to_numpy()))))
        .to_numpy()
    )
    group_df["mse"] = (
        df.groupby("scenario_type", dropna=False)
        .apply(lambda g: float(np.mean((g[pred_col].to_numpy() - g[tgt_col].to_numpy()) ** 2)))
        .to_numpy()
    )
    group_df["rmse"] = np.sqrt(group_df["mse"])
    group_df = group_df.sort_values("pred_risk_mean", ascending=False)
    return group_df


def summarize_risk_distribution(
    df: pd.DataFrame,
    column: str,
    bins: List[float] = None,
) -> Dict[str, Any]:
    values = df[column].to_numpy(dtype=np.float64)

    if bins is None:
        bins = [i / 10 for i in range(11)]

    hist_bins = [-np.inf] + bins + [np.inf]
    counts, edges = np.histogram(values, bins=hist_bins)

    intervals = []
    for i in range(len(counts)):
        left = edges[i]
        right = edges[i + 1]

        if np.isneginf(left):
            label = f"(-inf, {right:.2f})"
        elif np.isposinf(right):
            label = f"[{left:.2f}, +inf)"
        else:
            if i == len(counts) - 2:
                label = f"[{left:.2f}, {right:.2f}]"
            else:
                label = f"[{left:.2f}, {right:.2f})"

        intervals.append({
            "interval": label,
            "count": int(counts[i]),
            "ratio": float(counts[i] / len(values)) if len(values) > 0 else 0.0,
        })

    summary = {
        "column": column,
        "count": int(len(values)),
        "min": float(np.min(values)) if len(values) > 0 else np.nan,
        "max": float(np.max(values)) if len(values) > 0 else np.nan,
        "mean": float(np.mean(values)) if len(values) > 0 else np.nan,
        "std": float(np.std(values)) if len(values) > 0 else np.nan,
        "p01": float(np.percentile(values, 1)) if len(values) > 0 else np.nan,
        "p05": float(np.percentile(values, 5)) if len(values) > 0 else np.nan,
        "p25": float(np.percentile(values, 25)) if len(values) > 0 else np.nan,
        "p50": float(np.percentile(values, 50)) if len(values) > 0 else np.nan,
        "p75": float(np.percentile(values, 75)) if len(values) > 0 else np.nan,
        "p95": float(np.percentile(values, 95)) if len(values) > 0 else np.nan,
        "p99": float(np.percentile(values, 99)) if len(values) > 0 else np.nan,
        "intervals": intervals,
    }
    return summary


def print_risk_distribution(dist_summary: Dict[str, Any]):
    print(f"\n========= Risk Distribution: {dist_summary['column']} =========")
    print(f"count : {dist_summary['count']}")
    print(f"min   : {dist_summary['min']:.6f}")
    print(f"max   : {dist_summary['max']:.6f}")
    print(f"mean  : {dist_summary['mean']:.6f}")
    print(f"std   : {dist_summary['std']:.6f}")
    print(
        f"p01/p05/p25/p50/p75/p95/p99 : "
        f"{dist_summary['p01']:.6f} / "
        f"{dist_summary['p05']:.6f} / "
        f"{dist_summary['p25']:.6f} / "
        f"{dist_summary['p50']:.6f} / "
        f"{dist_summary['p75']:.6f} / "
        f"{dist_summary['p95']:.6f} / "
        f"{dist_summary['p99']:.6f}"
    )

    print("Interval distribution:")
    for item in dist_summary["intervals"]:
        print(
            f"{item['interval']:18s} | "
            f"count={item['count']:6d} | "
            f"ratio={item['ratio']:.4f}"
        )


@torch.no_grad()
def evaluate(data_loader, model, args, device):
    all_rows: List[Dict[str, Any]] = []
    running_offset = 0
    first_batch_debug = True

    calibrator = TeacherRiskCalibratorTorch(device=device)

    with tqdm(data_loader, desc="Evaluating RiskNet", unit="batch") as pbar:
        for batch in pbar:
            batch_size = infer_batch_size(batch)

            inputs, ego_future_gt, neighbors_future_gt = build_inputs_from_batch(batch, device)
            inputs = args.observation_normalizer(inputs)

            teacher = build_future_risk_teacher(
                ego_current_state=batch[0].to(device),
                ego_future_gt=ego_future_gt,
                neighbor_agents_past=batch[2].to(device),
                neighbors_future_gt=neighbors_future_gt,
                route_lanes=batch[7].to(device),
                route_lanes_speed_limit=batch[8].to(device),
                route_lanes_has_speed_limit=batch[9].to(device),
                static_objects=batch[10].to(device),
            )

            outputs = model(inputs)

            if first_batch_debug:
                print("\n========== Debug First Batch ==========")
                print("type(outputs):", type(outputs))
                if isinstance(outputs, dict):
                    print("outputs keys:", list(outputs.keys()))
                    for k, v in outputs.items():
                        if isinstance(v, torch.Tensor):
                            print(f"outputs[{k}] shape = {tuple(v.shape)}")
                print("type(teacher):", type(teacher))
                if isinstance(teacher, dict):
                    print("teacher keys:", list(teacher.keys()))
                    for k, v in teacher.items():
                        if isinstance(v, torch.Tensor):
                            print(f"teacher[{k}] shape = {tuple(v.shape)}")
                print("=======================================\n")
                first_batch_debug = False

            # 模型输出已经是CDF后的风险空间
            pred_risk_t = outputs["risk_score"].reshape(-1)[:batch_size]

            # teacher先取raw，再做CDF
            teacher_risk_raw_t = teacher["risk_target"].reshape(-1)[:batch_size]
            teacher_risk_cdf_t = calibrator.transform(teacher_risk_raw_t).reshape(-1)[:batch_size]

            pred_risk = pred_risk_t.detach().cpu().numpy()
            teacher_risk_cdf = teacher_risk_cdf_t.detach().cpu().numpy()

            pred_ttc = flatten_b1(outputs["ttc_proxy"])[:batch_size]
            teacher_ttc = flatten_b1(teacher["r_ttc"])[:batch_size]

            pred_dist = flatten_b1(outputs["dist_proxy"])[:batch_size]
            teacher_dist = flatten_b1(teacher["r_dist"])[:batch_size]

            scenario_types = resolve_scenario_type(batch, batch_size)
            scenario_tokens = resolve_scenario_token(batch, batch_size, running_offset)

            abs_err = np.abs(pred_risk - teacher_risk_cdf)
            sq_err = (pred_risk - teacher_risk_cdf) ** 2

            for i in range(batch_size):
                all_rows.append({
                    "scenario_token": scenario_tokens[i],
                    "scenario_type": scenario_types[i],

                    "pred_risk": float(pred_risk[i]),
                    "teacher_risk_cdf": float(teacher_risk_cdf[i]),
                    "abs_error": float(abs_err[i]),
                    "sq_error": float(sq_err[i]),

                    "pred_ttc": float(pred_ttc[i]),
                    "teacher_ttc": float(teacher_ttc[i]),

                    "pred_dist": float(pred_dist[i]),
                    "teacher_dist": float(teacher_dist[i]),
                })

            running_offset += batch_size

    return pd.DataFrame(all_rows)


def print_summary(summary: Dict[str, Any], title="Overall Metrics"):
    print(f"\n================ {title} ================")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"{k:20s}: {v:.6f}")
        else:
            print(f"{k:20s}: {v}")


def print_group_table(group_df: pd.DataFrame, title="Mean Risk by Scenario Type"):
    print(f"\n========= {title} =========")
    for _, row in group_df.iterrows():
        print(
            f"{str(row['scenario_type']):40s} | "
            f"count={int(row['count']):3d} | "
            f"pred_mean={row['pred_risk_mean']:.6f} | "
            f"teacher_mean={row['teacher_risk_mean']:.6f} | "
            f"mae={row['mae']:.6f}"
        )


def main():
    args = get_args()
    set_seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    test_set = RiskDiffuserTestData(
        args.test_set,
        args.test_set_list,
        args.agent_num,
        args.predicted_neighbor_num,
        args.future_len,
    )
    print(f"Dataset prepared: {len(test_set)} samples")

    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )

    model = load_risk_model(args, device)

    df = evaluate(test_loader, model, args, device)

    summary = summarize_global(df, pred_col="pred_risk", tgt_col="teacher_risk_cdf")
    group_df = summarize_by_scenario_type(df, pred_col="pred_risk", tgt_col="teacher_risk_cdf")

    pred_dist_summary = summarize_risk_distribution(df, "pred_risk")
    teacher_dist_summary = summarize_risk_distribution(df, "teacher_risk_cdf")

    print_summary(summary, title="Overall Metrics (Pred vs CDF-Teacher)")
    print_group_table(group_df, title="Mean Risk (Pred vs CDF-Teacher) by Scenario Type")

    print_risk_distribution(pred_dist_summary)
    print_risk_distribution(teacher_dist_summary)

    all_csv = os.path.join(args.output_dir, "risk_eval_all_samples.csv")
    group_csv = os.path.join(args.output_dir, "risk_eval_by_scenario_type.csv")
    summary_json = os.path.join(args.output_dir, "risk_eval_summary.json")

    pred_dist_json = os.path.join(args.output_dir, "risk_eval_pred_distribution.json")
    teacher_dist_json = os.path.join(args.output_dir, "risk_eval_teacher_cdf_distribution.json")

    dist_csv = os.path.join(args.output_dir, "risk_eval_distribution_bins.csv")

    df.to_csv(all_csv, index=False)
    group_df.to_csv(group_csv, index=False)

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4, ensure_ascii=False)

    with open(pred_dist_json, "w", encoding="utf-8") as f:
        json.dump(pred_dist_summary, f, indent=4, ensure_ascii=False)

    with open(teacher_dist_json, "w", encoding="utf-8") as f:
        json.dump(teacher_dist_summary, f, indent=4, ensure_ascii=False)

    dist_rows = []
    for item in pred_dist_summary["intervals"]:
        dist_rows.append({
            "type": "pred_risk",
            "interval": item["interval"],
            "count": item["count"],
            "ratio": item["ratio"],
        })
    for item in teacher_dist_summary["intervals"]:
        dist_rows.append({
            "type": "teacher_risk_cdf",
            "interval": item["interval"],
            "count": item["count"],
            "ratio": item["ratio"],
        })
    pd.DataFrame(dist_rows).to_csv(dist_csv, index=False)

    print("\nSaved files:")
    print(all_csv)
    print(group_csv)
    print(summary_json)
    print(pred_dist_json)
    print(teacher_dist_json)
    print(dist_csv)


if __name__ == "__main__":
    main()
