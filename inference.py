# Load image, preprocess, make POST request to the model server,
# then do post-processing on the inference (if necessary)
import tensorflow as tf
import numpy as np
import json
import requests

SIZE=224
MODEL_URI='http://localhost:8501/v1/models/face-mask-detection:predict'
CLASSES = ['mask', 'no-mask']

def __get_face_images(image_path):
    pass
def __draw_boxes(image_path):
    pass
    
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


