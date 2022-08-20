from PL_DataModules.neu_coco import NEUCOCODataModule
from PL_DataModules.gc10_coco import GC10COCODataModule
from PL_DataModules.al import ALCOCODataModule
from PL_DataModules.coco import COCODataModule
from PL_DataModules.voc import VOCDataModule


CONFIGS = {
    'neu_coco': NEUCOCODataModule,
    'gc10_coco': GC10COCODataModule,
    'al_coco': ALCOCODataModule,
    'coco': COCODataModule,
    'voc': VOCDataModule,
}


def build_data(model_name):
    return CONFIGS[model_name]
