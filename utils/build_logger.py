from pytorch_lightning.loggers import NeptuneLogger


def build_logger(logger, model, configs):
    if logger == "Neptune":
        neptune_logger = NeptuneLogger(
            project="chihaya/Steel",
            api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHV"
                      "uZS5haSIsImFwaV9rZXkiOiI5MGNmMzI2ZC1mOGYyLTQ2NzUtOTk0OS1kMmI3OGE1MTYwODQifQ==",
            log_model_checkpoints=False,
            tags=[configs["model"], "NEU-DET"],  # optional
        )
        neptune_logger.log_hyperparams(params=configs)
        neptune_logger.log_model_summary(model=model, max_depth=-1)
        return neptune_logger
    else:
        return True
