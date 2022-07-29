from PL_DataModules.neu_det import COCODataModule


CONFIGS = {
    'coco': COCODataModule,
}


def build_data(model_name):
    return CONFIGS[model_name]
