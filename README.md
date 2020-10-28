# face-mask-detection
Detecting face masks with CNNs using Tensorflow's Keras API. <br>
Dataset from [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset).<br>

## Objective
The goal of this project is to train a convolutional neural network which detects the presence of face masks in images.<br> Given an input image, the model will perform inference and output a prediction, a vector containing two probabilities -  P(mask) and P(no-mask). We will return the class that has highest probability.

## How to replicate
Download the dataset from [here](https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset).<br>

To process the data, run
```
python3 split.py <path to images> <path to csv file> <train %> <valid %> <test %>
```
The script uses the bounding box data provided by the Kaggle dataset (in train.csv) to crop the raw images and generates new images of faces. <br>The images are placed in a new *data/* directory under *train/* *valid/* or *test/* in either *mask/* or *no-mask/* based on the class information provided by train.csv. The last 3 arguments specify the split ratio for training/validation/test sets.<br><br>

To train the model and perform inference on test data, run
```
jupyter notebook
``` 
and open notebook *face_mask_detection_cnn.ipynb* or open in Google Colab to run the cells.

## Architecture
**Model from scratch** - 93% accuracy on test set
```
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_6 (Conv2D)            (None, 224, 224, 32)      896       
_________________________________________________________________
max_pooling2d_6 (MaxPooling2 (None, 112, 112, 32)      0         
_________________________________________________________________
conv2d_7 (Conv2D)            (None, 112, 112, 32)      9248      
_________________________________________________________________
max_pooling2d_7 (MaxPooling2 (None, 56, 56, 32)        0         
_________________________________________________________________
conv2d_8 (Conv2D)            (None, 56, 56, 64)        18496     
_________________________________________________________________
max_pooling2d_8 (MaxPooling2 (None, 28, 28, 64)        0         
_________________________________________________________________
flatten_2 (Flatten)          (None, 50176)             0         
_________________________________________________________________
dense_4 (Dense)              (None, 256)               12845312  
_________________________________________________________________
dropout_1 (Dropout)          (None, 256)               0         
_________________________________________________________________
dense_5 (Dense)              (None, 2)                 514       
=================================================================
Total params: 12,874,466
Trainable params: 12,874,466
Non-trainable params: 0
```

**Fine-tuned VGG16 model** - 98% accuracy on test set
```
Layer (type)                 Output Shape              Param #   
=================================================================
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792      
_________________________________________________________________
block1_conv2 (Conv2D)        (None, 224, 224, 64)      36928     
_________________________________________________________________
block1_pool (MaxPooling2D)   (None, 112, 112, 64)      0         
_________________________________________________________________
block2_conv1 (Conv2D)        (None, 112, 112, 128)     73856     
_________________________________________________________________
block2_conv2 (Conv2D)        (None, 112, 112, 128)     147584    
_________________________________________________________________
block2_pool (MaxPooling2D)   (None, 56, 56, 128)       0         
_________________________________________________________________
block3_conv1 (Conv2D)        (None, 56, 56, 256)       295168    
_________________________________________________________________
block3_conv2 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_conv3 (Conv2D)        (None, 56, 56, 256)       590080    
_________________________________________________________________
block3_pool (MaxPooling2D)   (None, 28, 28, 256)       0         
_________________________________________________________________
block4_conv1 (Conv2D)        (None, 28, 28, 512)       1180160   
_________________________________________________________________
block4_conv2 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_conv3 (Conv2D)        (None, 28, 28, 512)       2359808   
_________________________________________________________________
block4_pool (MaxPooling2D)   (None, 14, 14, 512)       0         
_________________________________________________________________
block5_conv1 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv2 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_conv3 (Conv2D)        (None, 14, 14, 512)       2359808   
_________________________________________________________________
block5_pool (MaxPooling2D)   (None, 7, 7, 512)         0         
_________________________________________________________________
flatten_15 (Flatten)         (None, 25088)             0         
_________________________________________________________________
dense_43 (Dense)             (None, 256)               6422784   
_________________________________________________________________
dropout_19 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_44 (Dense)             (None, 2)                 514       
=================================================================
Total params: 21,137,986
Trainable params: 21,137,986
Non-trainable params: 0
```

## Strategies for overfitting
* Augmented images to increase size of training set (horizontal flip, slight rotations).
* Used aggressive dropout regularization on Fully-connected layers. I noticed this helped reduce the gap between training and validation losses.
* Mess around with the number of layers/nodes to reduce overfitting the training data.

## Ideas for deployment
Build a basic web front-end which allows the user to take a picture using their webcam, and then classifies the image. The image can contain multiple persons, and the app should draw a colored bounding box around each person's face. The color will indicate the presence of a face mask detected by the machine learning model. Do the same for video (user can record a video, which will be processed and return a video which contains frames with colored bounding boxes.

## Resources
https://deeplizard.com/learn/video/DEMmkFC6IGM<br>
https://deeplizard.com/learn/video/ZjM_XQa5s6s<br>
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly<br>
https://medium.com/intelligentmachines/convolutional-neural-network-and-regularization-techniques-with-tensorflow-and-keras-5a09e6e65dc7<br>
https://cv-tricks.com/keras/fine-tuning-tensorflow/<br>
https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html
