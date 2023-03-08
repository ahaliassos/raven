import logging
import os

import hydra
from hydra.utils import instantiate
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.plugins import DDPPlugin
import torch

from data.data_module import DataModule
from finetune_learner import Learner
from utils import average_checkpoints


# static vars
os.environ["WANDB_SILENT"] = "true"
logging.getLogger("lightning").propagate = False


@hydra.main(config_path="conf", config_name="config_finetune")
def main(cfg):
    if cfg.fix_seed:
        seed_everything(42, workers=True)

    print("The SLURM job ID for this run is {}".format(os.environ["SLURM_JOB_ID"]))
    cfg.slurm_job_id = os.environ["SLURM_JOB_ID"]

    cfg.gpus = torch.cuda.device_count()
    print('num gpus:', cfg.gpus)
    
    wandb_logger = None
    if cfg.log_wandb:
        wandb_logger = instantiate(cfg.logger)

    data_module = DataModule(cfg)
    learner = Learner(cfg)

    ckpt_callback = ModelCheckpoint(
        monitor=cfg.checkpoint.monitor,
        mode=cfg.checkpoint.mode,
        dirpath=os.path.join(cfg.checkpoint.dirpath, cfg.experiment_name) if cfg.checkpoint.dirpath else None,
        save_last=True,
        filename=f'{{epoch}}',
        save_top_k=cfg.checkpoint.save_top_k,
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
        strategy = DDPPlugin(find_unused_parameters=False) if cfg.gpus > 1 else None
    )

    if cfg.train:
        trainer.fit(learner, data_module)

        # only 1 process should save the checkpoint and compute WER
        torch.distributed.destroy_process_group()
        if trainer.is_global_zero:
            last = [
                os.path.join(
                    cfg.checkpoint.dirpath, cfg.experiment_name, f"epoch={n}.ckpt"
                ) for n in range(trainer.max_epochs - cfg.model.avg_ckpts, trainer.max_epochs)
            ]
            avg = average_checkpoints(last)

            model_path = os.path.join(
                cfg.checkpoint.dirpath, cfg.experiment_name, f"model_avg_{cfg.model.avg_ckpts}.pth"
            )
            torch.save(avg, model_path)

            # compute WER
            if cfg.compute_final_wer:
                if cfg.test_on_one_gpu:
                    cfg.gpus = cfg.trainer.gpus = cfg.trainer.num_nodes = 1
                cfg.model.pretrained_model_path = model_path
                cfg.model.transfer_only_encoder = False
                cfg.change_ckpt_style = False
                data_module = DataModule(cfg)
                learner = Learner(cfg)
                trainer = Trainer(**cfg.trainer, logger=wandb_logger, strategy=None)
                trainer.test(learner, datamodule=data_module)
        
    else:
        trainer.test(learner, datamodule=data_module)


if __name__ == "__main__":
    main()
