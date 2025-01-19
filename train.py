import os

import hydra
import lightning.pytorch as pl
import torch
from omegaconf import DictConfig

from scripts.data import MyDataModule
from scripts.model import MyModel
from scripts.utils import logger_init


@hydra.main(config_path="conf", config_name="config_train", version_base="1.3")
def main(cfg: DictConfig):
    """
    Main training function for initializing and training the model using PyTorch Lightning.

    Args:
        cfg (DictConfig): Configuration object created using Hydra, containing all the necessary parameters
                          for data processing, model initialization, and training.

    Returns:
        None
    """
    # Set random seed for reproducibility
    pl.seed_everything(cfg.model.seed)

    # Initialize logger
    logger = logger_init(cfg.data.logger_name)
    logger.info("\n\n\n ### TRAIN NEW INSTANCE MODEL ###")

    torch.set_float32_matmul_precision(cfg.train.mantissa)

    # If you want to load data use this lines
    # start_date = datetime.strptime(cfg.data.start_date, "%Y-%m-%d").date()
    # end_date = datetime.strptime(cfg.data.end_date, "%Y-%m-%d").date()

    # Initialize data module with the given configuration and logger
    dm = MyDataModule(
        conf=cfg,
        logger=logger,
        # load_data=True,
        # start_date=start_date,
        # end_date=end_date,
        # out_file=f"./data/history_from_{cfg.data.start_date}_to_{end_date}.csv",
    )

    # Get statistics from the data module
    item2id, len_train_dataloader = dm.get_statistics()
    logger.info(f"Len item2id {item2id}")
    logger.info(f"Len train dataloader {len_train_dataloader}")

    # Initialize model with the given configuration, item-to-id len, logger, and dataloader length
    model = MyModel(
        conf=cfg,
        item2id=item2id,
        logger=logger,
        len_train_dataloader=len_train_dataloader,
        train_mode=True,
    )

    # Initialize model weights
    model.init_weights()
    logger.info(dm.val_dataset[0])

    # Configure loggers
    loggers = [
        pl.loggers.CSVLogger("logs/my-csv-logs", name=cfg.artifacts.experiment_name),
        pl.loggers.WandbLogger(
            project=cfg.artifacts.project,
            name=cfg.artifacts.experiment_name,
            config={
                "trim_length": cfg.train.trim_length,
                "mask_prob": cfg.model.mask_prob,
                "accumulation_steps": cfg.train.accumulation_steps,
                "min_seq_len": cfg.data.min_seq_len,
                "min_item_cnt": cfg.data.min_item_cnt,
            },
        ),  # лучше убрать
        pl.loggers.MLFlowLogger(
            experiment_name=cfg.artifacts.experiment_name,
            tracking_uri="http://10.66.8.151:13412",
        ),
    ]

    # Configure callbacks
    callbacks = [
        pl.callbacks.LearningRateMonitor(logging_interval="step"),
        # pl.callbacks.DeviceStatsMonitor(),
        # pl.callbacks.RichModelSummary(max_depth=cfg.callbacks.model_summary.max_depth),
    ]

    if cfg.callbacks.swa.use:
        callbacks.append(
            pl.callbacks.StochasticWeightAveraging(swa_lrs=cfg.callbacks.swa.lrs)
        )

    if cfg.artifacts.checkpoint.use:
        callbacks.append(
            pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(
                    cfg.artifacts.checkpoint.dirpath, cfg.artifacts.experiment_name
                ),
                filename=cfg.artifacts.checkpoint.filename,
                monitor=cfg.artifacts.checkpoint.monitor,
                save_top_k=cfg.artifacts.checkpoint.save_top_k,
                # every_n_train_steps=cfg.artifacts.checkpoint.every_n_train_steps,
                every_n_epochs=cfg.artifacts.checkpoint.every_n_epochs,
                mode="max",
                save_weights_only=True,
            )
        )

    # Initialize the PyTorch Lightning trainer with the given configuration
    trainer = pl.Trainer(
        accelerator=cfg.train.accelerator,
        devices=cfg.train.devices,
        precision=cfg.train.precision,
        max_epochs=cfg.train.num_epochs,
        # accumulate_grad_batches=cfg.train.grad_accum_steps,
        # val_check_interval=cfg.train.val_check_interval,
        overfit_batches=cfg.train.overfit_batches,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        deterministic=cfg.train.full_deterministic_mode,
        benchmark=cfg.train.benchmark,  # может ускорить, а может нет (данные константы по входу)
        gradient_clip_val=cfg.train.gradient_clip_val,
        profiler=cfg.train.profiler,
        log_every_n_steps=cfg.train.log_every_n_steps,
        detect_anomaly=cfg.train.detect_anomaly,  # хорошая штука для дебага модели
        enable_checkpointing=cfg.artifacts.checkpoint.use,
        logger=loggers,
        callbacks=callbacks,
    )

    # Optionally find the optimal batch size if the flag is set in the configuration
    if cfg.train.batch_size_finder:
        tuner = pl.tuner.Tuner(trainer)
        tuner.scale_batch_size(model, datamodule=dm, mode="power")
    # Start training the model
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
