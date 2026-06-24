import os
from torch.utils.tensorboard import SummaryWriter

try:
    import wandb
except ImportError:
    wandb = None

class TensorBoardLogger():
    def __init__(self, run_name, notes, args, wandb_resume_id, save_path, rank=0):
        """
        project_name (str): wandb project name
        config: dict or argparser
        """              
        self.args = args
        self.writer = None
        self.id = None
        
        if rank == 0:
            if args.use_wandb and wandb is not None:
                os.environ["WANDB_MODE"] = "online"
                wandb_writer = wandb.init(
                    project='RiskDiffuser',
                    name=run_name,
                    notes=notes,
                    resume="allow",
                    id=wandb_resume_id,
                    sync_tensorboard=True,
                    dir=f'{save_path}',
                )
                wandb.config.update(args)
                self.id = wandb_writer.id
            elif args.use_wandb:
                print("Warning: wandb is not installed; using TensorBoard only.")
	            
            self.writer = SummaryWriter(log_dir=f'{save_path}/tb')
    
    def log_metrics(self, metrics: dict, step: int):
       """
       metrics (dict):
       step (int, optional): epoch or step
       """
       if self.writer is not None:
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, step)

    def finish(self):
       if self.writer is not None:
            self.writer.close()
