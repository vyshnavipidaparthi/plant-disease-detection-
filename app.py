import streamlit as st
import tensorflow as tf
import numpy as np

# Tensorflow Model Prediction
def model_prediction(test_image):
    model_path = "B:\\PDC\\Plant_Disease_Prediction\\app\\models\\trained_plant_disease_model.keras"
    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("🌿 PlantGuard AI 🔍")
    image_path = "B:\\PDC\\Plant_Disease_Prediction\\app\\static\\uploads\\home_page.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    # 🌿 *Welcome to the Plant Disease Recognition System!* 🔍  
    ### *Protect Your Crops, Ensure a Healthier Harvest!*  

    Plants are the foundation of our food supply, and their health is vital. Our *AI-powered Plant Disease Recognition System* helps *farmers, gardeners, and researchers* detect plant diseases *quickly and accurately*—so you can take action before it’s too late!  

    ## 🌱 *How It Works*  
    🔹 *Upload an Image:* Visit the *Disease Recognition* page and upload a photo of your plant.  
    🔹 *AI-Powered Analysis:* Our advanced machine learning model scans the image to detect potential diseases.  
    🔹 *Get Instant Results:* Receive a *detailed diagnosis* along with expert recommendations for treatment and prevention.  

    ## 🌟 *Why Choose Us?*  
    ✅ *Cutting-Edge Accuracy:* Powered by *state-of-the-art deep learning* models for precise disease detection.  
    ✅ *Fast & Reliable:* Get results in *seconds—empowering you to act **immediately!*  
    ✅ *User -Friendly:* A *seamless and intuitive* experience, built for *farmers, researchers, and agricultural experts.*  
    ✅ *Proactive Crop Protection:* Stay *ahead of plant diseases* and maximize yield with *early detection.*  

    ## 🚀 *Get Started Now!*  
    Click on the *Disease Recognition* page in the sidebar, *upload your plant image, and let our **AI-powered system* do the rest!  

    ## 🌍 *About Us*  
    We are passionate about *revolutionizing plant health* with AI. Learn more about our *mission, team, and vision* on the *About* page.  

    🌾 *Together, let’s cultivate a healthier world—one plant at a time!* 🍃  
    """)

# About Project
elif app_mode == "About":
    st.header("🌿 PlantGuard AI 🔍")
    st.markdown("AI-Powered Plant Health Detection for a Sustainable Future")

    # Project Overview
    st.subheader("🌱 Project Overview")
    st.markdown("""
        PlantGuard AI is an advanced AI-driven plant disease detection system designed to assist farmers, researchers, 
        and agricultural professionals in identifying plant diseases quickly and accurately. By analyzing images of 
        plants, our system leverages deep learning to classify diseases in various crops, such as tomatoes, peppers, 
        and potatoes.
    """)

    # Project Goal
    st.subheader("🎯 Project Goal")
    st.markdown("""
        To provide a fast, reliable, and user-friendly solution for early disease detection, helping users take timely 
        preventive action and protect crop health.
    """)

    # Key Features
    st.subheader("✨ Key Features")
    st.markdown("""
        - ✅ State-of-the-Art AI Models – Trained on large datasets for high-accuracy disease detection.
        - ✅ Multi-Crop Support – Identifies diseases in tomatoes, peppers, potatoes, and more.
        - ✅ Simple & Intuitive Interface – Just upload an image, and our AI does the rest!
        - ✅ Optimized Model Selection – Choose between:
            - 🔹 High-Accuracy Model – Best for precision-focused detection.
            - 🔹 Balanced Model – Faster, optimized for real-time use.
        - ✅ Instant Results – Get a disease diagnosis with confidence scores in seconds.
    """)

    # About the Dataset
    st.subheader("📊 About the Dataset")
    st.markdown("""
        Our dataset is enhanced through offline augmentation and based on a publicly available dataset from GitHub. 
        It includes 87,000+ RGB images of healthy and diseased crop leaves, categorized into 38 distinct classes.
    """)
    st.subheader("Dataset Breakdown:")
    st.markdown("""
        - 📌 Training Set – 70,295 images (80%)
        - 📌 Validation Set – 17,572 images (20%)
        - 📌 Test Set – 33 images (for real-world evaluation)
    """)
    st.markdown("""
        This structured dataset ensures robust model performance for real-world applications.
    """)

    # How It Works
    st.subheader("🛠 How It Works")
    st.markdown("""
        Getting started is simple:
        1️⃣ Upload an image of the plant with suspected disease.

        2️⃣ Select a trained AI model for disease detection.

        3️⃣ The AI analyzes the image and predicts the disease with a confidence score.
        
        4️⃣ View results instantly, including disease name, confidence level, and recommendations.
    """)

    # Contact Us
    st.subheader("📩 Contact Us")
    st.markdown("""
        For inquiries, collaborations, or technical support, reach out to our team:
        - 📧 Email: [contact@plantguardai.com](mailto:contact@plantguardai.com)
        - 📞 Phone: +1 234 567 890
        - 📍 Address: 123 Plant Science Street, Agriculture City, 12345
    """)

    # Team Members
    st.subheader("👨‍💻 Meet the Team")
    st.markdown("""
        - 🔹 R. Durga Bhavani – 21KN1A42E6
        - 🔹 P. Vyshnavi – 21KN1A42D6
        - 🔹 T. Jyothika – 21KN1A42H6
        - 🔹 S. Teja – 22KN5A4215
    """)

    st.markdown("🌾 Together, we are committed to revolutionizing plant health with AI! 🚀")
# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=4, use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple__healthy',
                      'Blueberry__healthy', 'Cherry(including_sour)_Powdery_mildew',
                      'Cherry_(including_sour)healthy', 'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)Common_rust', 'Corn_(maize)Northern_Leaf_Blight', 'Corn(maize)_healthy',
                      'Grape__Black_rot', 'Grape_Esca(Black_Measles)', 'Grape__Leaf_blight(Isariopsis_Leaf_Spot)',
                      'Grape__healthy', 'Orange_Haunglongbing(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach__healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell__healthy',
                      'Potato__Early_blight', 'Potato_Late_blight', 'Potato__healthy',
                      'Raspberry__healthy', 'Soybean_healthy', 'Squash__Powdery_mildew',
                      'Strawberry__Leaf_scorch', 'Strawberry_healthy', 'Tomato__Bacterial_spot',
                      'Tomato__Early_blight', 'Tomato_Late_blight', 'Tomato__Leaf_Mold',
                      'Tomato__Septoria_leaf_spot', 'Tomato__Spider_mites Two-spotted_spider_mite',
                      'Tomato__Target_Spot', 'Tomato_Tomato_Yellow_Leaf_Curl_Virus', 'Tomato__Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))