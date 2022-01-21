import os
import yaml

with open('configs/detection/yolox_s.yaml', encoding='ascii', errors='ignore') as f:
    config = yaml.safe_load(f)  # model dict
    img_size = tuple(config['DATASET']['IMG_SIZE'])
    print(config)
