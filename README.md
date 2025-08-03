# 🌿 Plant Disease Diagnosis AI 🌿

A smart web application that uses deep learning to diagnose diseases in plant leaves from user-uploaded images. This project aims to provide a helpful tool for farmers and gardeners to quickly detect diseases, enabling timely intervention.

**✨ Live Demo: [https://huggingface.co/spaces/VietAnh05/plant-disease-app]**

---

### ## Key Features

* **Modern UI:** A clean, user-friendly, and responsive interface.
* **Flexible Image Upload:** Supports both clicking to select a file and drag-and-drop functionality.
* **Instant Preview:** Users can preview the selected image directly on the interface.
* **Real-Time Diagnosis:** Get predictions from the AI model within seconds.
* **Advanced Model Architecture:** Utilizes Transfer Learning with powerful architectures like EfficientNet and MobileNetV2 for high accuracy.
* **Smart Training:** Implements callbacks like `EarlyStopping` and `ReduceLROnPlateau` to optimize the training process.

---

### ## Dataset and Model Accuracy

The accuracy of the AI model is highly dependent on the quality and variety of the training data. This model was trained on the **PlantVillage dataset**, which includes a diverse range of crops.

The model is expected to perform with higher accuracy for the following types of plants included in the training set:
* Apple
* Blueberry
* Cherry
* Corn (Maize)
* Grape
* Orange
* Peach
* Pepper bell
* Potato
* Raspberry
* Soybean
* Squash
* Strawberry
* Tomato

For plants not included in this dataset, the model may not provide accurate predictions. The performance can be further improved by augmenting the dataset with more images and training for more epochs.

---

### ## Technology Stack

* **Backend:**
    * **Python:** The core programming language.
    * **Flask:** A lightweight web framework for building the API and serving the web app.
    * **TensorFlow / Keras:** The deep learning library for building and training the AI model.
    * **NumPy / OpenCV:** For image processing and numerical computations.

* **Frontend:**
    * **HTML5:** The structural foundation of the web page.
    * **CSS3:** For styling and visual effects.
    * **JavaScript (ES6):** To handle user interactions and asynchronous image uploads via the `fetch` API.

---

### ## Local Setup and Installation

To run this project on your local machine, follow these steps:

#### 1. Environment Setup

```bash
# 1. Clone this repository
git clone https://github.com/EthanNguyen1412/Plant_Disease_Diagnosis_AI.git
cd plant-disease-app

# 2. Create a virtual environment (recommended)
python -m venv venv
# On Windows: venv\Scripts\activate
# On macOS/Linux: source venv/bin/activate

# 3. Install the required libraries
pip install -r requirements.txt
```

#### 2. Model Training (Optional)

If you wish to train the model yourself, prepare the **PlantVillage dataset** in a `dataset` directory and run the following command:
```bash
python train_model.py
```
*Note: This process can be very time-consuming.*

#### 3. Running the Web Server

Ensure you have the trained model file (`.h5`) and the `class_names.json` file. Update the model path in `app.py` if necessary.
```bash
# Run the Flask server
flask run
```
Then, open your web browser and navigate to `http://127.0.0.1:5000`.

---
### ## Project Structure
```
plant-disease-app/
├── static/              # Stores uploaded images
├── templates/
│   └── index.html       # The user interface
├── app.py               # The Flask web server
├── train_model.py       # The AI model training script
├── plant_disease...h5   # The trained model file
├── class_names.json     # The disease class names
└── requirements.txt     # A list of required libraries
```
