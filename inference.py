# Load image, preprocess, make POST request to the model server,
# then do post-processing on the inference (if necessary)
import sys
import requests
import json
import cv2
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from PIL import Image

SIZE=224
MODEL_URI='http://localhost:8501/v1/models/face-mask-detection:predict'
CLASSES = ['mask', 'no-mask']

# Given predictions, draw bounding boxes on image and save
def __draw_boxes(image_path, result_list):


# Given an image, detect and return face bounding boxes
def __detect_faces(image_path):
    # load detector
    detector = MTCNN()
    filename = image_path
    # load image from file
    pixels = plt.imread(filename)
    # detect faces in image
    faces = detector.detect_faces(pixels)
    # return value for each face, including bounding box
    return faces

# Given an image object, bounding box coords (list), return cropped image
def __crop_image(image, bounding_box):
    # get Pillow box tuple : left, upper, right, lower
    left, upper = bounding_box[0], bounding_box[1]
    right = bounding_box[0] + bounding_box[2]
    lower = bounding_box[1] + bounding_box[3]
    
    box = (left, upper, right, lower)
    # Crop Pillow image
    img = image.crop(box)
    return img

# Given 
# Gets prediction for each face, draws colored bounding boxes
# saves image file, returns path to file on disk
def get_prediction(image_path):
    image = tf.keras.preprocessing.image.load_img(
        image_path, target_size=(SIZE, SIZE)
    )

    # convert image to numpy array
    image = tf.keras.preprocessing.image.img_to_array(image)
    # normalize image with same technique used during training
    image = tf.keras.applications.vgg16.preprocess_input(image)
    # add batch dimension to obtain (1, 224, 224, 3) tensor
    image = np.expand_dims(image, axis=0)

    # json object to send to model server
    data = json.dumps({
        'instances': image.tolist()
    })

    # UTF-8 encoding
    response = requests.post(MODEL_URI, data=data.encode())
    result = json.loads(response.text)

    # output of softmax activation function
    prediction = np.squeeze(result['predictions'][0][0])
    class_name = CLASSES[int(prediction < 0.5)]
    return class_name

def get_preds(image_path):
    # get bounding boxes for each face
    faces = __detect_faces(image_path)

    # get image object using Pillow
    img = Image.open(image_path)

    imgs = []
    # get cropped image for each face
    for face in faces:
        bounding_box = face.get('box')
        imgs.append(__crop_image(img, bounding_box))
        
    # preprocess images
    imgs_pre = []
    for i in imgs:
        i = i.resize((SIZE, SIZE), Image.ANTIALIAS)
        # convert image to numpy array
        i = tf.keras.preprocessing.image.img_to_array(i)
        # normalize image using same technique as training
        i = tf.keras.applications.vgg16.preprocess_input(i)
        # add batch dimension
        i = np.expand_dims(i, axis=0)
        # add to preprocessed tensor list
        imgs_pre.append(i)

    # make predictions
    preds = []

    tensors = []
    for i in imgs_pre:
        tensors.append(i.tolist())

     # json object to send to model server
    data = json.dumps({
        'instances': tensors
    })

    # UTF-8 encoding
    response = requests.post(MODEL_URI, data=data.encode())
    result = json.loads(response.text)
    print(type(result))
    print(result)
    # output of softmax activation function
    #prediction = np.squeeze(result['predictions'][0][0])
    #class_name = CLASSES[int(prediction < 0.5)]
   

    


