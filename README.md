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


.
+-- _config.yml
+-- _drafts
|   +-- begin-with-the-crazy-ideas.textile
|   +-- on-simplicity-in-technology.markdown
+-- _includes
|   +-- footer.html
|   +-- header.html
+-- _layouts
|   +-- default.html
|   +-- post.html
+-- _posts
|   +-- 2007-10-29-why-every-programmer-should-play-nethack.textile
|   +-- 2009-04-26-barcamp-boston-4-roundup.textile
+-- _data
|   +-- members.yml
+-- _site
+-- index.html
