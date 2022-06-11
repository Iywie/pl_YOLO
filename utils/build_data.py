from PL_DataModules.neu_det import NEUDataModule


CONFIGS = {
    'neu-det': NEUDataModule,
}


def build_data(model_name):
    return CONFIGS[model_name]
