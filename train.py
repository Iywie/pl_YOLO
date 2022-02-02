import torch
import pytorch_lightning
from pytorch_lightning import Trainer
import torch.backends.cudnn as cudnn

from utils.defaults import default_argument_parser
from utils.config_file import merge_config

from detection.yolox import LitYOLOX


def main():
    pytorch_lightning.seed_everything(96)
    configs = merge_config(args.cfg)
    print("Command Line Configs:", configs)
    model = LitYOLOX(configs)

    # parameters:
    # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.trainer.trainer.Trainer.html?highlight=trainer
    trainer = Trainer(
        # Train
        # devices="cpu",
        benchmark=False,
        check_val_every_n_epoch=10,
        # Experiment
        # default_root_dir="lightning_logs",
        log_every_n_steps=50,
        # Debug
        # limit_train_batches=2,
        # limit_val_batches=1,
        detect_anomaly=False,
    )

    trainer.fit(model, )
    # trainer.validate(model)
    # trainer.test(model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main()
