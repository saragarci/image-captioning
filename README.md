[//]: # (Image References)

[image1]: ./images/image-captioning.png "Image Captioning Model"

# Image Captioning

This project implements a neural network architecture to automatically generate captions from images.

The Microsoft Common Objects in COntext [(MS COCO) dataset](https://cocodataset.org/#home) is used to train the network.


![Image Captioning Model][image1]


This project is broken down into four main notebooks:

* Notebook 0: Exploring the dataset
* Notebook 1: Preliminaries
* Notebook 2: Defining and Training a the network
* Notebook 3: Inferece

## Local Environment Instructions

1. Clone this repo: https://github.com/cocodataset/cocoapi  
```
git clone https://github.com/cocodataset/cocoapi.git  
```

2. Setup the coco API (also described in the readme [here](https://github.com/cocodataset/cocoapi)) 
```
cd cocoapi/PythonAPI  
make  
cd ..
```

3. Download some specific data from here: http://cocodataset.org/#download (described below)

* Under **Annotations**, download:
  * **2014 Train/Val annotations [241MB]** (extract captions_train2014.json and captions_val2014.json, and place at locations cocoapi/annotations/captions_train2014.json and cocoapi/annotations/captions_val2014.json, respectively)  
  * **2014 Testing Image info [1MB]** (extract image_info_test2014.json and place at location cocoapi/annotations/image_info_test2014.json)

* Under **Images**, download:
  * **2014 Train images [83K/13GB]** (extract the train2014 folder and place at location cocoapi/images/train2014/)
  * **2014 Val images [41K/6GB]** (extract the val2014 folder and place at location cocoapi/images/val2014/)
  * **2014 Test images [41K/6GB]** (extract the test2014 folder and place at location cocoapi/images/test2014/)

## Credits

### Resources

[Starting project](https://github.com/udacity/CVND---Image-Captioning-Project). 


### Contributors

* [Sara Garci](s@saragarci.com)
* [Computer Vision Nanodegree](https://www.udacity.com/course/computer-vision-nanodegree--nd891)

## License

© Copyright 2021 by Sara Garci. All rights reserved.
