from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import DeviceStatsMonitor
from pytorch_lightning.loggers import NeptuneLogger
from utils.defaults import argument_parser, load_config
from utils.build_model import build_model


def main():
    args = argument_parser().parse_args()
    configs = load_config(args.cfg)

    model = build_model(configs['model'])
    model = model(configs)

    neptune_logger = NeptuneLogger(
        project="chihaya/YOLOX",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHV"
                  "uZS5haSIsImFwaV9rZXkiOiI5MGNmMzI2ZC1mOGYyLTQ2NzUtOTk0OS1kMmI3OGE1MTYwODQifQ==",
        log_model_checkpoints=False,
        tags=["YOLOX", "NEU-DET"],  # optional
    )
    neptune_logger.log_hyperparams(params=configs)
    neptune_logger.log_model_summary(model=model, max_depth=-1)

    device_stats = DeviceStatsMonitor()

    # Trainer parameters:
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer
    trainer = Trainer(
        # tpu_cores=8,
        gpus=1,
        # amp_backend="apex",
        # amp_level=01,
        # auto_lr_find=True,
        # benchmark=False,
        check_val_every_n_epoch=5,
        # callbacks=[device_stats],
        # default_root_dir="lightning_logs",
        # detect_anomaly=True,
        enable_progress_bar=True,
        # logger=neptune_logger,
        log_every_n_steps=10,
        # limit_train_batches=3,
        # limit_val_batches=0,
        max_epochs=100,
        # reload_dataloaders_every_n_epochs=10,
    )

    trainer.fit(model)
    # trainer.validate(model)
    # trainer.test(model)


if __name__ == "__main__":
    main()
