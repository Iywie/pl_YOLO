import pytorch_lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor

from utils.defaults import default_argument_parser
from utils.config_file import merge_config

from detection.yolox import LitYOLOX


def main():
    pytorch_lightning.seed_everything(96)
    device_stats = DeviceStatsMonitor()
    configs = merge_config(args.cfg)
    print("Command Line Configs:", configs)
    model = LitYOLOX(configs)

    # parameters:
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer
    trainer = Trainer(
        # tpu_cores=8,
        # gpus=1,
        # amp_backend="apex",
        # amp_level=01,
        # auto_lr_find=True,
        # benchmark=False,
        check_val_every_n_epoch=10,
        callbacks=[device_stats],
        # default_root_dir="lightning_logs",
        detect_anomaly=True,
        # enable_progress_bar=False,
        log_every_n_steps=50,
        limit_train_batches=2,
        limit_val_batches=1,
        max_epochs=300,
        # reload_dataloaders_every_n_epochs=10,
        # resume_from_checkpoint='',
    )

    trainer.fit(model)
    # trainer.validate(model)
    # trainer.test(model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main()
