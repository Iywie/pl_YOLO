from PL_DataModules.coco import COCODataModule
from PL_DataModules.voc import VOCDataModule


CONFIGS = {
    'coco': COCODataModule,
    'voc': VOCDataModule,
}


def build_data(model_name):
    return CONFIGS[model_name]
