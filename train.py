import torch
import pytorch_lightning
from pytorch_lightning import Trainer
import torch.backends.cudnn as cudnn

from utils.defaults import default_argument_parser
from utils.config_file import merge_config

from detection.yolox import LitYOLOX


def main():
    pytorch_lightning.seed_everything(96)
    cudnn.benchmark = True
    configs = merge_config(args.cfg)
    print("Command Line Configs:", configs)
    model = LitYOLOX(configs)
    torch.autograd.set_detect_anomaly(True)

    trainer = Trainer(limit_train_batches=2, limit_val_batches=1)

    trainer.fit(model, )
    # trainer.validate(model)
    # trainer.test(model)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    main()
