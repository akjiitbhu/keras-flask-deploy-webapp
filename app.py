from ctypes import sizeof
import os
import sys
import cv2

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect, send_file
import joblib
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from datetime import datetime
import io
from PIL import Image



# Some utilites
import numpy as np
from util import base64_to_pil, np_img


# Declare a flask app
app = Flask(__name__)


# model

image_size = 256   

def down_block(x, filters, kernel_size=(5, 5), padding="same", strides=1):  
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation('relu')(c)
    res = keras.layers.Conv2D(filters, (1, 1), strides=(2, 2), padding='same')(c)
    res = keras.layers.BatchNormalization()(res)
    p = keras.layers.MaxPool2D((2, 2), (2, 2))(c)
    x = keras.layers.add([p, res])
    return c, x

def up_block(x, skip, filters, kernel_size=(5, 5), padding="same", strides=1):
    us = keras.layers.UpSampling2D((2, 2))(x)
    concat = keras.layers.Concatenate()([us, skip])
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(concat)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation('relu')(c)
    return c

def bottleneck(x, filters, kernel_size=(5, 5), padding="same", strides=1):
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation('relu')(c)
    c = keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(c)
    c = keras.layers.BatchNormalization()(c)
    c = keras.layers.Activation('relu')(c)

    return c

def UNet():
    f = [16, 32, 64, 128, 256]
    inputs = keras.layers.Input((image_size, image_size, 3))
    
    p0 = inputs
    c1, p1 = down_block(p0, f[0]) #128 -> 64
    c2, p2 = down_block(p1, f[1]) #64 -> 32
    c3, p3 = down_block(p2, f[2]) #32 -> 16
    c4, p4 = down_block(p3, f[3]) #16->8
    
    bn = bottleneck(p4, f[4])
    
    u1 = up_block(bn, c4, f[3]) #8 -> 16
    u2 = up_block(u1, c3, f[2]) #16 -> 32
    u3 = up_block(u2, c2, f[1]) #32 -> 64
    u4 = up_block(u3, c1, f[0]) #64 -> 128
    
    outputs = keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(u4)
    model = keras.models.Model(inputs, outputs)
    return model

model = UNet()
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["acc"])



print('Model loaded. Check http://127.0.0.1:5002/')


# Model saved with Keras model.save()
MODEL_PATH = 'models/LocaliDeepDO_14_02.h5'

# Load your own trained model
model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
print('Model loaded. Start serving...')


def model_predict(img, model):
    # Preprocessing the image
    #img = cv2.imread(img, 1)
    x = img.resize((256, 256))
    x = image.img_to_array(x)
    x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)
    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='tf')
    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get the image from post request
        img = base64_to_pil(request.json)
        img_name = str(datetime.now())
        img.save("./uploads/"+img_name+".png")
        # Save the image to ./uploads

        # Make prediction
        preds = model_predict(img, model)
        n_result = preds > 0.5
        #result = np_img(n_result[0])

        # Process your result for human
        #pred_proba = "{:.3f}".format(np.amax(preds))    # Max probability
        #pred_class = decode_predictions(preds, top=1)   # ImageNet Decode

        #result = str(pred_class[0][0][1])               # Convert to string
        #result = result.replace('_', ' ').capitalize()
        
        # Serialize the result, you can add additional fields
        mask = img_name+"_mask.png"
        
        cv2.imwrite("./uploads/mask/"+mask,n_result[0]*255)
        img_r = cv2.imread("./uploads/mask/"+mask,0)
        return send_file(img_r, mimetype='image/PNG')
    return None


if __name__ == '__main__':
    app.run(port=5009, threaded=False)

    # Serve the app with gevent
    #http_server = WSGIServer(('0.0.0.0', 5000), app)
    #http_server.serve_forever()
