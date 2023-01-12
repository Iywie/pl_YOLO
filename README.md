# pl-YOLO
Detection on pytorch lightning.

## Step of training
### 1. Build the model yaml and data yaml in configs.
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
In configs/data/coco2017.yaml   
Change the directory, train size and batch size.
```
    dir: <path to your COCO2017>
```

#### Run YOLOX-s
```python
  python train.py -d configs/data/coco2017.yaml -c configs/model/yolox/yolox_s.yaml
```
