# 🌱 Plant Disease Detection

A machine learning project that detects plant diseases from images using deep learning techniques.  
The project also includes a simple website interface for easy image upload and disease prediction.


## 📌 Features
- Preprocessing and augmentation of plant leaf images
- Deep learning model training and evaluation
- Disease classification with accuracy metrics
- Web interface for real-time predictions
- Modular and extensible codebase


## 🗂️ Project Structure
```

Plant\_disease\_detection/
├── notebooks/             # Jupyter notebooks for model training and Source code(python scripts)
│   └── Train\_plant\_disease.ipynb
├── app/                   # Website frontend (HTML/CSS/JS or React)
│   ├── index.html
│   ├── style.css
│   └── app.js
├── requirements.txt       # Python dependencies
├── README.md              # Project documentation
└── .gitignore             # Ignore datasets, models, etc.


## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/vyshnavipidaparthi/Plant_disease_detection.git
   cd Plant_disease_detection
````

2. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux/Mac
   venv\Scripts\activate      # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Usage

### 🔹 Train the Model

```bash
python src/train.py
```

### 🔹 Make Predictions

```bash
python src/predict.py --image path/to/leaf.jpg
```

### 🔹 Run the Web App

If using a simple static site:

```bash
open web/index.html
```

If using Flask/Django (backend required):

```bash
python app.py
```


## 📊 Results

* Achieved **99% accuracy** on validation dataset
* Supports multiple plant species and disease categories
* Confusion matrix and training curves available in `/notebooks`


## 🌍 Applications

* Assist farmers in early disease detection
* Reduce pesticide misuse by targeting specific diseases
* Enhance crop yield and food security


## 🔮 Future Improvements

* Improve accuracy with ResNet/EfficientNet
* Add support for more plant species


## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss.
Don’t forget to ⭐ the repo if you find it useful.


## 📜 License

This project is licensed under the college conference. 
