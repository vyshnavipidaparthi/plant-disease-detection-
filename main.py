from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import os
import joblib

app = Flask(__name__)

# Load the models once when the app starts
cnn_model = tf.keras.models.load_model("B:/PDC/Plant_Disease_Prediction/app/models/cnn_feature_extractor_model.keras")
svm_model = joblib.load("B:/PDC/Plant_Disease_Prediction/app/models/svm_plant_disease_model.pkl")

# TensorFlow Model Prediction
def model_prediction(test_image):
    # Load and preprocess the image
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch

    # Extract features using the CNN model
    features = cnn_model.predict(input_arr)
    features_flattened = features.reshape(features.shape[0], -1)  # Flatten the features

    # Make prediction using the SVM model
    prediction_index = svm_model.predict(features_flattened)
    return prediction_index[0]  # Return the predicted class index

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/disease_recognition', methods=['GET', 'POST'])
def disease_recognition():
    prediction = None
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Save the uploaded image temporarily
            image_path = os.path.join('static', file.filename)
            file.save(image_path)
            # Make prediction
            result_index = model_prediction(image_path)
            class_names = [
                'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                'Tomato___healthy'
            ]
            prediction = class_names[result_index]
            os.remove(image_path)  # Clean up the uploaded image after prediction
    return render_template('disease_recognition.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)