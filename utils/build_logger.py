from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.loggers import CSVLogger


def build_logger(logger, model, configs, version):
    if logger == "nep":
        neptune_logger = NeptuneLogger(
            project="chihaya/YOLO-Series",
            name=version,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdH"
                      "VuZS5haSIsImFwaV9rZXkiOiI5MGNmMzI2ZC1mOGYyLTQ2NzUtOTk0OS1kMmI3OGE1MTYwODQifQ==",
            log_model_checkpoints=False,
            tags=[configs["model"], "NEU-DET"],  # optional
        )
        neptune_logger.log_hyperparams(params=configs)
        neptune_logger.log_model_summary(model=model, max_depth=-1)
        return neptune_logger
    if logger == 'csv':
        csv_logger = CSVLogger("logs", name="csvlogger", version=version)
        csv_logger.log_hyperparams(params=configs)
        return csv_logger
    if logger == 'cn':
        csv_logger = CSVLogger("logs", name="csvlogger", version=version)
        csv_logger.log_hyperparams(params=configs)

        neptune_logger = NeptuneLogger(
            project="chihaya/YOLO-Series",
            name=version,
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdH"
                      "VuZS5haSIsImFwaV9rZXkiOiI5MGNmMzI2ZC1mOGYyLTQ2NzUtOTk0OS1kMmI3OGE1MTYwODQifQ==",
            log_model_checkpoints=False,
            tags=[configs["model"], "NEU-DET"],  # optional
        )
        neptune_logger.log_hyperparams(params=configs)
        neptune_logger.log_model_summary(model=model, max_depth=-1)
        return csv_logger, neptune_logger

    else:
        return True
