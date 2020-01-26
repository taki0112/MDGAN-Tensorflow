## MDGAN &mdash; Simple TensorFlow Implementation [[Paper]](https://arxiv.org/abs/1903.05854)
### : Learning Text-to-image Generation by Redescription

<div align="center">
  <img src="./assets/teaser.png">
</div>

## Requirements
* **tensorflow >= 1.15**

## Dataset
### Text
* [birds](https://drive.google.com/file/d/1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ/view)
* [coco](https://drive.google.com/file/d/1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9/view)

### Image
* [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html)
* [coco](http://cocodataset.org/#download)

## Usage
```
├── dataset
   └── YOUR_DATASET_NAME
       ├── images
           ├── domain1 (domain folder)
               ├── xxx.jpg (domain1 image)
               ├── yyy.png
               ├── ...
           ├── domain2
               ├── aaa.jpg (domain2 image)
               ├── bbb.png
               ├── ...
           ├── domain3
           ├── ...
       ├── text
           ├── captions.pickle
           ├── filenames_train.pickle
           ├── filenames_test.pickle
```

### Train
```
python main.py --dataset birds --phase train
```

### Test
```
python main.py --dataset birds --phase test
```



## Author
[Junho Kim](http://bit.ly/jhkim_ai)
