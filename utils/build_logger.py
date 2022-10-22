import os
import time
from pytorch_lightning.loggers import NeptuneLogger, CSVLogger, TensorBoardLogger, WandbLogger


def build_logger(logger, project_name, name, model, configs):

    timestamp = time.strftime('%Y%m%d_%H%M', time.localtime())
    save_dir = os.path.join('./log', f'{timestamp+name}')

    if logger == 'csv':
        csv_logger = CSVLogger(save_dir=save_dir, name="csvlogger", version=timestamp)
        csv_logger.log_hyperparams(params=configs)
        return csv_logger

    if logger == 'wdb':
        if not os.path.exists("./log"):
            os.mkdir("./log")
        wandb_logger = WandbLogger(project=project_name, save_dir="log", version=name)
        return wandb_logger

    if logger == "nep":
        neptune_logger = NeptuneLogger(
            project="chihaya/YOLO-Series",
            name=name,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdH"
                      "VuZS5haSIsImFwaV9rZXkiOiI5MGNmMzI2ZC1mOGYyLTQ2NzUtOTk0OS1kMmI3OGE1MTYwODQifQ==",
            log_model_checkpoints=False,
            tags=[configs["model"], "NEU-DET"],  # optional
        )
        neptune_logger.log_hyperparams(params=configs)
        neptune_logger.log_model_summary(model=model, max_depth=-1)
        return neptune_logger

    # Default: tensorboard
    else:
        tensorboard = TensorBoardLogger(save_dir='logs', name=name, version=timestamp)
        tensorboard.log_hyperparams(params=configs)
        return tensorboard
