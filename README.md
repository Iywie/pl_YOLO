# pl-YOLO
Detection on pytorch lightning.

## Step of training
### 1. Build the model configuration and data configuratioin in configs.
### 2. Run
```python
  python train.py -c <path to the model config> -d <path to the data config>
```

## Example
### COCO2017 Dataset
COCO2017   
└───annotations_trainval2017   
|   └───annotations     
|   |   |   instances_train2017.json   
|   |   |   instances_val2017.json   
└───train2017    
└───val2017   


```
COCO2017  
│
└───annotations_trainval2017   
│   └───annotations   
│       │   instances_train2017.json   
│       │   instances_train2017.json   
└───train2017
└───val2017 
```
