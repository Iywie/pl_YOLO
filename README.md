# pl-YOLO
Detection on pytorch lightning.

## Step of training
### 1. Build the model parameters and dataset parameters in configs.
### 2. Run
```python
  python train.py -c <path to the model yaml> -d <path to the data yaml>
```

## Example
### COCO2017 Dataset
#### Dataset directory
```
COCO2017  
└─annotations_trainval2017   
│   └─annotations   
│      │ instances_train2017.json   
│      │ instances_train2017.json   
└─train2017
└─val2017 
```
#### Change dataset parameters
In configs/data/coco2017.yaml, change the directory, image size and batch size.
```
    dir: <path to your COCO2017>
    ...
    train_size: [640,640]
    val_size: [640,640]
    train_batch_size: 32
    val_batch_size: 32
```

#### Run YOLOX-s
```python
  python train.py -d configs/data/coco2017.yaml -c configs/model/yolox/yolox_s.yaml
```
