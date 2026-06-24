# RiskDiffuser

Official implementation for:

> Qian Y, Wang Z, Wu Y, et al. **RiskDiffuser: Continuous risk conditioning for diffusion planning in autonomous driving**. *Expert Systems with Applications*, 2026: 133115.

RiskDiffuser is a diffusion-based planner for autonomous driving with continuous risk conditioning. The planner receives a continuous risk level and learns to generate future ego trajectories whose behavior can be adjusted by risk. This repository also includes a RiskNet module that predicts scene-level risk from the same encoded scene representation and can be used to provide risk conditioning automatically.

## Overview

The codebase contains:

- `riskdiffuser/model/riskdiffuser.py`: RiskDiffuser planner model.
- `riskdiffuser/model/risknet.py`: RiskNet risk prediction head and future-risk teacher.
- `riskdiffuser/planner/planner.py`: nuPlan planner wrapper.
- `train_predictor.py`: train or fine-tune the RiskDiffuser planner.
- `train_risk_net.py`: train RiskNet from a pretrained RiskDiffuser encoder.
- `test_risk_net.py`: evaluate RiskNet predictions against the future-risk teacher.
- `data_process.py`: preprocess nuPlan scenarios into `.npz` training samples.
- `sim_riskdiffuser_runner.sh`: run closed-loop nuPlan simulation.

## Installation

Create a Python environment and install the required packages:

```bash
pip install -r requirements_torch.txt
pip install -e .
```

The code was developed with Python 3.9 and PyTorch 2.8.0. nuPlan simulation requires a working nuPlan devkit installation and the nuPlan dataset.

## Data Preparation

Edit the paths in `data_process.sh`:

```bash
NUPLAN_DATA_PATH="REPLACE_WITH_NUPLAN_DATA_DIR"
NUPLAN_MAP_PATH="REPLACE_WITH_NUPLAN_MAPS_DIR"
TRAIN_SET_PATH="REPLACE_WITH_PREPROCESSED_TRAIN_SET_DIR"
```

Then run:

```bash
bash data_process.sh
```

The training scripts expect:

- `--train_set`: directory containing preprocessed `.npz` files.
- `--train_set_list`: JSON file containing the relative `.npz` file list.

Several shell scripts use `REPLACE_WITH_...` placeholders. Replace them with your local paths before running.

## Checkpoints

Place pretrained checkpoints under:

```text
checkpoints/
  args.json
  latest.pth
  riskcheckpoint/
    args.json
    latest.pth
```

Model weights are not included in this repository. Please refer to `checkpoint.txt` for download instructions, then place the downloaded weights in the corresponding checkpoint directory.

## Train RiskNet

Edit dataset paths in `torch_run_risk.sh`, then run:

```bash
bash torch_run_risk.sh
```

The script trains RiskNet using a pretrained RiskDiffuser checkpoint:

```bash
python train_risk_net.py \
  --train_set REPLACE_WITH_PREPROCESSED_TRAIN_SET_DIR \
  --train_set_list REPLACE_WITH_TRAIN_SET_LIST_JSON \
  --pretrained_riskdiffuser_ckpt checkpoints/ \
  --freeze_encoder true \
  --device cuda
```

## Train or Fine-Tune RiskDiffuser

Edit dataset paths in `torch_run.sh`, then run:

```bash
bash torch_run.sh
```

The script loads the planner checkpoint from `checkpoints/` and the RiskNet checkpoint from `checkpoints/riskcheckpoint/`.

## Evaluate RiskNet

Edit test set paths in `torch_test_risk.sh`, then run:

```bash
bash torch_test_risk.sh
```

The evaluation writes:

- `risk_eval_all_samples.csv`
- `risk_eval_by_scenario_type.csv`
- `risk_eval_summary.json`
- risk distribution summaries

## nuPlan Simulation

Edit nuPlan paths in `sim_riskdiffuser_runner.sh`:

```bash
NUPLAN_DEVKIT_ROOT="REPLACE_WITH_NUPLAN_DEVKIT_ROOT"
NUPLAN_DATA_ROOT="REPLACE_WITH_NUPLAN_DATA_ROOT"
NUPLAN_MAPS_ROOT="REPLACE_WITH_NUPLAN_MAPS_DIR"
NUPLAN_EXP_ROOT="REPLACE_WITH_NUPLAN_EXP_DIR"
```

Then run:

```bash
bash sim_riskdiffuser_runner.sh
```

The script uses the planner configuration in `riskdiffuser/config/planner/riskdiffuser.yaml` and passes the local checkpoints in `checkpoints/`.

## NuBoard

`run_nuboard.ipynb` provides a notebook template for viewing nuPlan simulation results. Replace every `REPLACE_WITH_...` value in the notebook before launching NuBoard.

## Citation

If this work is useful for your research, please cite:

```bibtex
@article{qian2026riskdiffuser,
  title = {RiskDiffuser: Continuous risk conditioning for diffusion planning in autonomous driving},
  author = {Qian, Yao and Wang, Zhiling and Wu, Yanfei and Wang, Yuxiang},
  journal = {Expert Systems with Applications},
  volume = {331},
  pages = {133115},
  year = {2026},
  issn = {0957-4174},
  doi = {10.1016/j.eswa.2026.133115}
}
```

## License

Please add the license terms for this repository before public release.
