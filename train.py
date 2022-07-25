from pytorch_lightning import Trainer, seed_everything
from utils.defaults import train_argument_parser, load_config
from utils.build_model import build_model
from utils.build_data import build_data
from utils.build_logger import build_logger


def main():
    # Read argument and model configs
    args = train_argument_parser().parse_args()

    configs = load_config(args.cfg)
    model = build_model(configs['model'])
    model = model(configs)

    if args.data is not None:
        configs['dataset']['dir'] = args.data
    data = build_data(configs['datamodule'])
    data = data(configs)

    logger = build_logger(args.logger, model, configs, args.version)

    seed_everything(96, workers=True)

    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        max_epochs=300,
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        enable_progress_bar=True,
        logger=logger,
        # callbacks=[timer],
        # precision=16,
        # amp_backend="apex",
        # amp_level=01,
        # auto_lr_find=True,
        # benchmark=False,
        # default_root_dir="lightning_logs",
        # detect_anomaly=True,
        limit_train_batches=80,
        limit_val_batches=40,
        # reload_dataloaders_every_n_epochs=10,
    )

    trainer.fit(model, datamodule=data)
    # trainer.tune(model, datamodule=data)
    # trainer.validate(model, datamodule=data, ckpt_path='lightning_logs/version_0/checkpoints/epoch=44-step=4050.ckpt')
    # trainer.test(model, datamodule=data)


if __name__ == "__main__":
    main()
