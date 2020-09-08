# face-mask-detection
Detecting face masks with CNNs using Tensorflow's Keras API. <br>
Dataset from [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset).<br>

## Objective
The goal of this project is to train a convolutional neural network which detects the presence of face masks in images.<br> Given an input image, the model will perform inference and output a prediction, which is a one-hot encoded vector containing two probabilities -  P(mask) and P(no-mask). We will return the class that has highest probability (see [argmax](https://machinelearningmastery.com/argmax-in-machine-learning/#:~:text=Argmax%20is%20an%20operation%20that,function%20is%20preferred%20in%20practice)).
## How to run
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

## Resources
https://deeplizard.com/learn/video/DEMmkFC6IGM<br>
https://deeplizard.com/learn/video/ZjM_XQa5s6s<br>
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly<br>
https://danijar.com/tips-for-training-recurrent-neural-networks/
https://www.researchgate.net/post/When_can_Validation_Accuracy_be_greater_than_Training_Accuracy_for_Deep_Learning_Models
https://medium.com/intelligentmachines/convolutional-neural-network-and-regularization-techniques-with-tensorflow-and-keras-5a09e6e65dc7
