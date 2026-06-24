
import io
import os
import random
import warnings
import torch
import numpy as np
from typing import Deque, Dict, List, Type
from timm.utils import ModelEma
from mmengine import fileio

warnings.filterwarnings("ignore")

_PLANNER_MODEL_CACHE = {}
_RISK_MODEL_CACHE = {}

from nuplan.common.actor_state.ego_state import EgoState
from nuplan.common.utils.interpolatable_state import InterpolatableState
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from nuplan.planning.simulation.trajectory.abstract_trajectory import AbstractTrajectory
from nuplan.planning.simulation.trajectory.interpolated_trajectory import InterpolatedTrajectory
from nuplan.planning.simulation.observation.observation_type import Observation, DetectionsTracks
from nuplan.planning.simulation.planner.ml_planner.transform_utils import transform_predictions_to_states
from nuplan.planning.simulation.planner.abstract_planner import (
    AbstractPlanner, PlannerInitialization, PlannerInput
)

from riskdiffuser.model.riskdiffuser import RiskDiffuser as RiskDiffuserModel
from riskdiffuser.model.risknet import SharedEncoderRiskNet
from riskdiffuser.data_process.data_processor import DataProcessor
from riskdiffuser.utils.config import Config


def identity(ego_state, predictions):
    return predictions


def _resolve_ckpt_path(path: str) -> str:
    if os.path.isdir(path):
        for name in ["latest.pth", "model.pth"]:
            candidate = os.path.join(path, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(f"No checkpoint found in directory: {path}")
    return path


class RiskDiffuserPlanner(AbstractPlanner):
    def __init__(
            self,
            config: Config,
            ckpt_path: str,

            past_trajectory_sampling: TrajectorySampling, 
            future_trajectory_sampling: TrajectorySampling,

            enable_ema: bool = True,
            device: str = "cpu",
            risk_mode: str = "manual",
            manual_risk_level: float = 0.9,
            risk_model_path: str = None,
            risk_model_enable_ema: bool = True,
        ):

        assert device in ["cpu", "cuda"], f"device {device} not supported"
        if device == "cuda":
            assert torch.cuda.is_available(), "cuda is not available"
        assert risk_mode in ["manual", "risk_net"], f"risk_mode {risk_mode} not supported"
            
        self._future_horizon = future_trajectory_sampling.time_horizon # [s] 
        self._step_interval = future_trajectory_sampling.time_horizon / future_trajectory_sampling.num_poses # [s]
        
        self._config = config
        self._ckpt_path = ckpt_path

        self._past_trajectory_sampling = past_trajectory_sampling
        self._future_trajectory_sampling = future_trajectory_sampling

        self._ema_enabled = enable_ema
        self._device = device
        self._risk_mode = risk_mode
        self._manual_risk_level = manual_risk_level
        self._risk_model_path = risk_model_path
        self._risk_model_enable_ema = risk_model_enable_ema
        self._risk_model = None

        self._planner = RiskDiffuserModel(config)

        self.data_processor = DataProcessor(config)
        
        self.observation_normalizer = config.observation_normalizer

    def name(self) -> str:
        """
        Inherited.
        """
        return "riskdiffuser"
    
    def observation_type(self) -> Type[Observation]:
        """
        Inherited.
        """
        return DetectionsTracks

    def initialize(self, initialization: PlannerInitialization) -> None:
        """
        Inherited.
        """
        #####################################
        # CUR_SEED = 0

        # random.seed(CUR_SEED)
        # np.random.seed(CUR_SEED)
        # torch.manual_seed(CUR_SEED)

        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        #####################################

        self._map_api = initialization.map_api
        self._route_roadblock_ids = initialization.route_roadblock_ids
        self._planner = self._get_or_load_planner_model()

        if self._risk_mode == "risk_net":
            self._risk_model = self._get_or_load_risk_model()

        self._initialization = initialization

    def _get_or_load_planner_model(self):
        cache_key = (self._ckpt_path, self._device, self._ema_enabled)
        if cache_key in _PLANNER_MODEL_CACHE:
            return _PLANNER_MODEL_CACHE[cache_key]

        planner = self._planner.to(self._device)

        if self._ckpt_path is not None:
            state_dict: Dict = torch.load(self._ckpt_path, map_location=self._device, weights_only=False)
            if 0: #self._ema_enabled:
                state_dict = state_dict['ema_state_dict']
            else:
                if "model" in state_dict.keys():
                    state_dict = state_dict['model']

            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith("module."):
                    new_state_dict[k[7:]] = v
                else:
                    new_state_dict[k] = v

            planner.load_state_dict(new_state_dict, strict=False)
            print(f"Planner model loaded from {self._ckpt_path}")
        else:
            print("load random model")

        planner.eval()
        _PLANNER_MODEL_CACHE[cache_key] = planner
        return planner

    def _load_frozen_risk_model(self):
        if self._risk_model_path is None:
            raise ValueError("risk_mode='risk_net' requires risk_model_path to be set")

        base_model = RiskDiffuserModel(self._config).to(self._device)
        risk_model = SharedEncoderRiskNet(base_model.encoder, self._config).to(self._device)

        ckpt_path = _resolve_ckpt_path(self._risk_model_path)
        ckpt_bytes = fileio.get(ckpt_path)
        with io.BytesIO(ckpt_bytes) as f:
            ckpt = torch.load(f, map_location=self._device, weights_only=False)

        state_dict = ckpt["model"] if "model" in ckpt else ckpt
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
        msg = risk_model.load_state_dict(state_dict, strict=False)

        print(f"RiskNet loaded from {ckpt_path}")
        print("RiskNet missing keys:", msg.missing_keys)
        print("RiskNet unexpected keys:", msg.unexpected_keys)

        if self._risk_model_enable_ema and "ema_state_dict" in ckpt:
            ema = ModelEma(risk_model, decay=0.999, device=self._device)
            ema_state = ckpt["ema_state_dict"]
            if isinstance(ema_state, dict) and "module" in ema_state:
                ema_state = ema_state["module"]
            ema_state = {k.replace("module.", ""): v for k, v in ema_state.items()}
            try:
                ema.ema.load_state_dict(ema_state, strict=False)
                risk_model = ema.ema
                print("RiskNet EMA load done")
            except Exception as e:
                print(f"RiskNet EMA load failed, fallback to raw model. Reason: {e}")

        risk_model.eval()
        for param in risk_model.parameters():
            param.requires_grad_(False)

        return risk_model

    def _get_or_load_risk_model(self):
        cache_key = (
            self._risk_model_path,
            self._device,
            self._risk_model_enable_ema,
        )
        if cache_key in _RISK_MODEL_CACHE:
            return _RISK_MODEL_CACHE[cache_key]

        risk_model = self._load_frozen_risk_model()
        _RISK_MODEL_CACHE[cache_key] = risk_model
        return risk_model

    def _get_risk_level(self, normalized_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if self._risk_mode == "manual":
            return torch.tensor([[self._manual_risk_level]], dtype=torch.float32, device=self._device)

        with torch.no_grad():
            risk_outputs = self._risk_model(normalized_inputs)
            risk_level = risk_outputs["risk_score"].detach()
            #risk_level = 1- risk_level
        return risk_level

    def planner_input_to_model_inputs(self, planner_input: PlannerInput) -> Dict[str, torch.Tensor]:
        history = planner_input.history
        traffic_light_data = list(planner_input.traffic_light_data)
        model_inputs = self.data_processor.observation_adapter(history, traffic_light_data, self._map_api, self._route_roadblock_ids, self._device)

        normalized_inputs = self.observation_normalizer(model_inputs)
        normalized_inputs["risk_level"] = self._get_risk_level(normalized_inputs)
        return normalized_inputs

    def outputs_to_trajectory(self, outputs: Dict[str, torch.Tensor], ego_state_history: Deque[EgoState]) -> List[InterpolatableState]:    

        predictions = outputs['prediction'][0, 0].detach().cpu().numpy().astype(np.float64) # T, 4
        heading = np.arctan2(predictions[:, 3], predictions[:, 2])[..., None]
        predictions = np.concatenate([predictions[..., :2], heading], axis=-1) 

        states = transform_predictions_to_states(predictions, ego_state_history, self._future_horizon, self._step_interval)

        return states
    
    def compute_planner_trajectory(self, current_input: PlannerInput) -> AbstractTrajectory:
        """
        Inherited.
        """
        inputs = self.planner_input_to_model_inputs(current_input)
        _, outputs = self._planner(inputs)

        trajectory = InterpolatedTrajectory(
            trajectory=self.outputs_to_trajectory(outputs, current_input.history.ego_states)
        )

        return trajectory
    
