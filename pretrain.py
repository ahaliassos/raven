import logging
import os

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
import torch

from data.data_module import DataModule
from pretrain_learner import Raven

# static vars
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("lightning").propagate = False


@hydra.main(config_path="conf", config_name="config_pretrain")
def main(cfg):
    if cfg.fix_seed:
        seed_everything(42, workers=True)

    print("The SLURM job ID for this run is {}".format(os.environ["SLURM_JOB_ID"]))
    cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]

    cfg.gpus = torch.cuda.device_count()
    print("num gpus:", cfg.gpus)
    if cfg.gpus < 2:
        cfg.trainer.strategy = None

    wandb_logger = None
    if cfg.log_wandb:
        wandb_logger = instantiate(cfg.logger)

    data_module = DataModule(cfg)
    learner = Raven(cfg)

    ckpt_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        dirpath=os.path.join(cfg.checkpoint.dirpath, cfg.experiment_name)
        if cfg.checkpoint.dirpath
        else None,
        save_last=True,
        filename=f"{cfg.experiment_name}-{{epoch}}",
    )
    callbacks = []
    if cfg.log_wandb:
        callbacks = [
            ckpt_callback,
            LearningRateMonitor(logging_interval=cfg.logging.logging_interval),
        ]
    trainer = Trainer(
        **cfg.trainer,
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(learner, data_module)

    torch.distributed.destroy_process_group()

    # Save video and audio encoders

    path = os.path.join(cfg.checkpoint.dirpath, cfg.experiment_name, "last.ckpt")
    ckpt = torch.load(path)["state_dict"]
    ckpt_video = {
        k[22:]: v
        for k, v in ckpt.items()
        if k.startswith("model.model_video.backbone.encoder")
        and not k.endswith(("gamma_ff_macaron", "gamma_conv"))
    }
    ckpt_audio = {
        k[22:]: v
        for k, v in ckpt.items()
        if k.startswith("model.model_audio.backbone.encoder")
        and not k.endswith(("gamma_ff_macaron", "gamma_conv"))
    }

    target_path_video = os.path.join(
        cfg.checkpoint.dirpath, cfg.experiment_name, "video.pth"
    )
    os.makedirs(os.path.dirname(target_path_video), exist_ok=True)
    torch.save(ckpt_video, target_path_video)

    target_path_audio = os.path.join(
        cfg.checkpoint.dirpath, cfg.experiment_name, "audio.pth"
    )
    os.makedirs(os.path.dirname(target_path_audio), exist_ok=True)
    torch.save(ckpt_audio, target_path_audio)


if __name__ == "__main__":
    main()
