# face-mask-detection
Detecting face masks with CNNs using Tensorflow's Keras API. <br>
Dataset from [Kaggle Face Mask Detection Dataset](https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset).<br>

## Objective
The goal of this project is to detect the presence of face masks in images.
## How to run
Download the dataset from [here](https://www.kaggle.com/wobotintelligence/face-mask-detection-dataset).<br>

```
python3 split.py <path to images> <path to csv file> <train %> <valid %> <test %>
```
This script uses the bounding box data provided by the Kaggle dataset (in train.csv) to crop the raw images and generates new images of faces. <br>The images are placed in a new *data/* directory under *train/*, *valid/*, and *test/*.<br><br>
To train the model and perform inference on test data, run `jupyter notebook` and open the *face_mask_detection_cnn.ipynb* to run the cells.

## Resources
https://deeplizard.com/learn/video/YRhxdVk_sIs<br>
https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly<br>
https://danijar.com/tips-for-training-recurrent-neural-networks/
https://www.researchgate.net/post/When_can_Validation_Accuracy_be_greater_than_Training_Accuracy_for_Deep_Learning_Models
