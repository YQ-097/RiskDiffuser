import gc
import os
import argparse
import json
import multiprocessing as mp
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed

from riskdiffuser.data_process.data_processor import DataProcessor

from nuplan.planning.utils.multithreading.worker_parallel import SingleMachineParallelExecutor
from nuplan.planning.scenario_builder.scenario_filter import ScenarioFilter
from nuplan.planning.scenario_builder.nuplan_db.nuplan_scenario_builder import NuPlanScenarioBuilder

from tqdm import tqdm
import time as time_module


# =========================
# Scenario filter
# =========================
def get_filter_parameters(
    num_scenarios_per_type=None,
    limit_total_scenarios=None,
    shuffle=True,
    scenario_tokens=None,
    log_names=None,
):
    scenario_types = None
    map_names = None
    timestamp_threshold_s = None
    ego_displacement_minimum_m = None

    expand_scenarios = True
    remove_invalid_goals = False

    ego_start_speed_threshold = None
    ego_stop_speed_threshold = None
    speed_noise_tolerance = None

    return (
        scenario_types,
        scenario_tokens,
        log_names,
        map_names,
        num_scenarios_per_type,
        limit_total_scenarios,
        timestamp_threshold_s,
        ego_displacement_minimum_m,
        expand_scenarios,
        remove_invalid_goals,
        shuffle,
        ego_start_speed_threshold,
        ego_stop_speed_threshold,
        speed_noise_tolerance,
    )


# =========================
# batch helper
# =========================
def chunk_list(lst, chunk_size):
    for i in range(0, len(lst), chunk_size):
        yield lst[i : i + chunk_size]


# =========================
# db prepare / cleanup
# =========================
def prepare_db_files(token_chunk, windows_db_dir, wsl_db_dir):
    """
    从 D 盘复制当前 batch 所需的 db 文件到 WSL
    """
    log_names = sorted({t.split("/")[0] for t in token_chunk})
    print(f"  Preparing {len(log_names)} db files")

    for log in log_names:
        src = os.path.join(windows_db_dir, f"{log}.db")
        dst = os.path.join(wsl_db_dir, f"{log}.db")

        if not os.path.exists(src):
            raise FileNotFoundError(f"[DB MISSING] {src}")

        if not os.path.exists(dst):
            shutil.copy2(src, dst)


def cleanup_db_files(wsl_db_dir):
    """
    删除 WSL 目录下所有 db 文件
    """
    removed = 0
    for f in os.listdir(wsl_db_dir):
        if f.endswith(".db"):
            os.remove(os.path.join(wsl_db_dir, f))
            removed += 1
    print(f"  Cleaned {removed} db files")


# =========================
# 多进程 worker（必须是顶层函数）
# =========================
def process_one_scenario(scenario):
    try:
        DataProcessor.process_scenario(
            scenario,
            num_past_poses=20,
            past_time_horizon=2,
            num_agents=ARGS.agent_num,
            num_static=ARGS.static_objects_num,
            max_ped_bike=10,
            map_features=['LANE', 'LEFT_BOUNDARY', 'RIGHT_BOUNDARY', 'ROUTE_LANES'],
            radius=100,
            max_elements={
                'LANE': ARGS.lane_num,
                'LEFT_BOUNDARY': ARGS.lane_num,
                'RIGHT_BOUNDARY': ARGS.lane_num,
                'ROUTE_LANES': ARGS.route_num,
            },
            max_points={
                'LANE': ARGS.lane_len,
                'LEFT_BOUNDARY': ARGS.lane_len,
                'RIGHT_BOUNDARY': ARGS.lane_len,
                'ROUTE_LANES': ARGS.route_len,
            },
            num_future_poses=80,
            future_time_horizon=8,
            save_dir=ARGS.save_path,
        )
    except Exception as e:
        print(f"[ERROR] scenario {scenario.token} failed: {e}")
        raise


# =========================
# main
# =========================
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="NuPlan Data Processing (Batch DB Copy)")

    parser.add_argument("--data_path", type=str, default="REPLACE_WITH_NUPLAN_DATA_DIR")
    parser.add_argument("--map_path", type=str, default="REPLACE_WITH_NUPLAN_MAPS_DIR")
    parser.add_argument("--save_path", type=str, default="./cache")

    # db paths
    parser.add_argument("--windows_db_dir", type=str, default="REPLACE_WITH_WINDOWS_NUPLAN_DB_DIR")
    parser.add_argument("--wsl_db_dir", type=str, default="REPLACE_WITH_NUPLAN_DATA_DIR")

    # scenario
    parser.add_argument("--scenarios_per_type", type=int, default=None)
    parser.add_argument("--total_scenarios", type=int, default=None)
    parser.add_argument("--shuffle_scenarios", type=bool, default=True)

    # agent / map config
    parser.add_argument("--agent_num", type=int, default=32)
    parser.add_argument("--static_objects_num", type=int, default=5)

    parser.add_argument("--lane_len", type=int, default=20)
    parser.add_argument("--lane_num", type=int, default=70)

    parser.add_argument("--route_len", type=int, default=20)
    parser.add_argument("--route_num", type=int, default=25)

    # parallel
    parser.add_argument("--batch_size", type=int, default=10000)
    parser.add_argument("--num_workers", type=int, default=None)

    ARGS = parser.parse_args()
    os.makedirs(ARGS.save_path, exist_ok=True)

    # -------------------------
    # load scenario tokens
    # -------------------------
    with open("./nuplan_train.json", "r") as f: #nuplan_train_scenario_tokens
        scenario_tokens = json.load(f)

    print(f"Total scenario tokens: {len(scenario_tokens)}")

    # -------------------------
    # nuPlan builder
    # -------------------------
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    builder = NuPlanScenarioBuilder(
        data_root=ARGS.data_path,
        map_root=ARGS.map_path,
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
    )

    num_workers = ARGS.num_workers or min(4, mp.cpu_count())
    print(f"Using {num_workers} worker processes")

    total_processed = 0

# -------------------------
# batch loop（batch-isolated）
# -------------------------
for batch_idx, token_chunk in enumerate(
    chunk_list(scenario_tokens, ARGS.batch_size)
):
    print(f"\n========== Batch {batch_idx + 1} ==========")

    # ① copy db（用 log/token）
    prepare_db_files(
        token_chunk,
        ARGS.windows_db_dir,
        ARGS.wsl_db_dir,
    )

    # ② 只保留 / 后面的 scenario token
    scenario_only_tokens = [t.split("/", 1)[1] for t in token_chunk]

    print(f"[Batch {batch_idx + 1}] loading {len(scenario_only_tokens)} scenarios")

    # ===== 每个 batch 新建 worker + builder =====
    worker = SingleMachineParallelExecutor(use_process_pool=True)

    builder = NuPlanScenarioBuilder(
        data_root=ARGS.data_path,
        map_root=ARGS.map_path,
        sensor_root=None,
        db_files=None,
        map_version="nuplan-maps-v1.0",
    )

    scenario_filter = ScenarioFilter(
        *get_filter_parameters(
            scenario_tokens=scenario_only_tokens,
            shuffle=ARGS.shuffle_scenarios,
        )
    )

    scenarios = builder.get_scenarios(scenario_filter, worker)

    print(f"[Batch {batch_idx + 1}] start parallel processing")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_one_scenario, scenario)
            for scenario in scenarios
        ]

        for _ in tqdm(
            as_completed(futures),
            total=len(futures),
            desc=f"Batch {batch_idx + 1}",
        ):
            pass

    total_processed += len(scenarios)
    print(
        f"[Batch {batch_idx + 1}] processed {len(scenarios)} scenarios "
        f"(total={total_processed})"
    )

    # ===== 强制释放 batch 内所有状态 =====
    del scenarios, scenario_filter, builder, worker
    gc.collect()

    # ③ cleanup db（安全：builder 已销毁）
    cleanup_db_files(ARGS.wsl_db_dir)

    time_module.sleep(10)
