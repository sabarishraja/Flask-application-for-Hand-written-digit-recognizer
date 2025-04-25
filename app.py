from flask import Flask, render_template, request, redirect, url_for
import numpy as np
import tensorflow as tf 
from tf_keras.models import load_model
from PIL import Image
import cv2 

#Initializing the Flask application
app = Flask(__name__)

#Load the trained model 
model = load_model('mnist_model.h5')

@app.route('/')
def home():
    return render_template('index.html', prediction = None)

@app.route('/predict', methods = ['POST'])
def predict():
    #Condition to check if the image was uploaded
    if 'file' not in request.files:
        return redirect(request.url)
    
    #Get the uploaded file
    file = request.files['file']
    #If the filename is empty then redirect it
    if file.filename == '':
        return redirect(request.url)
    
    #Now, I have to read the image and pre-process it similar to the mnist images
    image = Image.open(file).convert('L') #Converting the image to grayscale
    image = image.resize((28, 28)) # Resizing the image
    image = np.array(image)
    image = image.reshape(1, 28, 28, 1) #Reshapin gthe model input
    image = image.astype('float32') / 255.0 #Normalizing the pixel values

    #Now, predicting the digit using the model
    prediction = model.predict(image)
    digit = np.argmax(prediction)

    return render_template('index.html', prediction = digit)

if __name__ == "__main__":
    app.run(debug=True)