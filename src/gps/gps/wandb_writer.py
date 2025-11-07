# wandb_writer.py
import os
from gps import ExperimentConfig

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    wandb = None
    _WANDB_AVAILABLE = False


class WandBWriter:
    def __init__(self, cfg: ExperimentConfig):
        if not _WANDB_AVAILABLE:
            raise ImportError("W&B not installed. Install using: pip install wandb")
        
        run_dir = os.path.join(cfg.log_dir, cfg.name)

        self.run = wandb.init(
            project='SS-GNN',
            name=cfg.name,
            config=cfg.parameter_dict(),
            dir=run_dir,
            tags=[cfg.dataset_name, cfg.model_name, cfg.model_config.mpnn_type]
        )

    def add_scalar(self, tag, value, step=None):
        wandb.log({tag: value}, step=step)

    def add_text(self, tag, text, step=None):
        wandb.log({tag: wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def add_histogram(self, tag, values, step=None):
        wandb.log({tag: wandb.Histogram(values)}, step=step)

    def add_image(self, tag, img, step=None):
        wandb.log({tag: wandb.Image(img)}, step=step)

    def watch(self, model, log="all", log_freq=200):
        wandb.watch(model, log=log, log_freq=log_freq)

    def close(self):
        wandb.finish()
