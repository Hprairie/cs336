import wandb

from argparse import Namespace

class WandbLogger:
    def __init__(self, cfg: Namespace) -> None:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=cfg.wandb_name, config=vars(cfg))
    
    def log(self, data: dict | str, step: int | None = None) -> None:
        if isinstance(data, dict):
            assert step is not None, "Attempting to log data to wandb without having a step set"
            wandb.log(data=data, step=step)
        elif isinstance(data, str):
            print(data)
        else:
            raise ValueError(f"Data passed to logger is of unknown type {type(data)}. Please use dict/str")