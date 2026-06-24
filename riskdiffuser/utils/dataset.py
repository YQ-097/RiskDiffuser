import os
from torch.utils.data import Dataset
import torch
import numpy as np

from riskdiffuser.utils.train_utils import openjson, opendata


import numpy as np
import sys
import matplotlib.pyplot as plt

RHO_CACHE = []
MAX_RHO_SAMPLES = 10000

def compute_risk_from_distance(
    ego_future,
    neighbors_future,
    sigma=8.0,
    clip_max=30.0
):

    ego_future = np.asarray(ego_future, dtype=np.float32)
    neighbors_future = np.asarray(neighbors_future, dtype=np.float32)

    # 没有邻居
    if neighbors_future.size == 0:
        return 0.0

    ego_xy = ego_future[:, :2]
    nei_xy = neighbors_future[:, :, :2]

    dist = np.linalg.norm(nei_xy - ego_xy[None, :, :], axis=-1)

    min_dist = np.min(dist)
    min_dist = np.clip(min_dist, None, clip_max)

    rho = np.exp(-min_dist / sigma)
    rho = rho ** 2

    return float(rho)


class RiskDiffuserData(Dataset):
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

        #ego_current_state = data['ego_current_state']
        #x,y,cos,sin,vx,vy,ax,ay,steering_angle,yaw_rate
        ego_current_state = data['ego_current_state'][..., :10]
        ego_agent_future = data['ego_agent_future']

        neighbor_agents_past = data['neighbor_agents_past'][:self._past_neighbor_num]
        neighbor_agents_future = data['neighbor_agents_future'][:self._predicted_neighbor_num]

        lanes = data['lanes']
        lanes_speed_limit = data['lanes_speed_limit']
        lanes_has_speed_limit = data['lanes_has_speed_limit']

        route_lanes = data['route_lanes']
        route_lanes_speed_limit = data['route_lanes_speed_limit']
        route_lanes_has_speed_limit = data['route_lanes_has_speed_limit']

        static_objects = data['static_objects']

        rho = compute_risk_from_distance(
            ego_agent_future,
            neighbor_agents_future
        )
        #print(rho)
        ###分布绘制
        # RHO_CACHE.append(rho)

        # if len(RHO_CACHE) >= MAX_RHO_SAMPLES:
        #     rhos = np.array(RHO_CACHE)

        #     print("===== RHO STATISTICS =====")
        #     print("Count:", len(rhos))
        #     print("Min:", rhos.min())
        #     print("Max:", rhos.max())
        #     print("Mean:", rhos.mean())
        #     print("Std:", rhos.std())

        #     # 画图
        #     plt.figure()
        #     plt.hist(rhos, bins=50)
        #     plt.xlabel("rho")
        #     plt.ylabel("frequency")
        #     plt.title("Rho Distribution")
        #     plt.show()

        #     print("Finished collecting rho samples. Exiting program.")
        #     sys.exit(0)

        data = {
            "ego_current_state": ego_current_state,
            "ego_future_gt": ego_agent_future,
            "neighbor_agents_past": neighbor_agents_past,
            "neighbors_future_gt": neighbor_agents_future,
            "lanes": lanes,
            "lanes_speed_limit": lanes_speed_limit,
            "lanes_has_speed_limit": lanes_has_speed_limit,
            "route_lanes": route_lanes,
            "route_lanes_speed_limit": route_lanes_speed_limit,
            "route_lanes_has_speed_limit": route_lanes_has_speed_limit,
            "static_objects": static_objects,
            "risk_level": np.array([rho], dtype=np.float32),  # 关键
        }

        return tuple(data.values())