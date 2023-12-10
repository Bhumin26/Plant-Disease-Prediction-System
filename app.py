from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
#from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask import jsonify


# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'Plants-Disease-Detection-using-Tensorflow-and-OpenCV-main/Deployment/models/plant_disease_prediction_model .h5'

# Load your trained model
model = load_model(MODEL_PATH)

#model._make_predict_function()      
print('Model loaded. Start serving...')

print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    
    #update by ViPS
    img = cv2.imread(img_path)
    new_arr = cv2.resize(img,(150,150))
    new_arr = np.array(new_arr/255)
    new_arr = new_arr.reshape(-1, 150, 150, 3)
    print("Shape of new_arr before reshaping:", new_arr.shape)
    

    
    preds = model.predict(new_arr)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/team',)
def about():
    return render_template('team.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads',f.filename )  #secure_filename(f.filename)
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)

        # Process your result for human
        pred_class = preds.argmax()              # Simple argmax
 
        
        CATEGORIES = ['Pepper_bell_Bacterial_spot','Pepper_bell_healthy',
            'Potato_Early_blight' ,'Potato_healthy','Potato_Late_blight',
            'Tomato_Bacterial_spot' ,'Tomato_Early_blight','Tomato_healthy',
            'Tomato_Late_blight']
    #     return jsonify({"prediction": CATEGORIES[pred_class]})
    # return jsonify({"error": "No file uploaded"})
        return CATEGORIES[pred_class]
    return None



if __name__ == '__main__':
    app.run(debug=True)