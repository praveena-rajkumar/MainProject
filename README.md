
# An AI-Driven Plant Disease Detection System with Organic Remedy

An AI-driven plant disease detection system designed for farmers to diagnose crop infections in real time using leaf images and provide safe, eco-friendly organic remedies. The system utilizes a Convolutional Neural Network (CNN) built with TensorFlow/Keras to achieve up to 90% diagnostic accuracy.

## 📌 Project Overview
* **Student Name:** Praveena
* **Batch:** TN-DS-AN B02
* **Project Category:** Deep Learning / CNN
* **Target Users:** Farmers

---

## 🎯 Problem Statement & Objectives
### Real-World Problem
Manual diagnosis of plant diseases is slow and prone to errors, leading to crop failure or excessive chemical pesticide use. This project resolves these issues by analyzing smartphone or drone images via a CNN for instant identification and providing non-toxic treatment suggestions.

### Objectives
* **Image Capture:** Process user photos of sick or discolored plant leaves.
* **Feature Extraction:** Scan for visual patterns, colors, and textures linked to pathogens.
* **Classification:** Query traits against a trained dataset to accurately name the disease.
* **Remedy Formulation:** Deliver a tailored organic treatment plan based on the class prediction.

---

## 📊 Dataset & Features
### Source
The model trains on the [Mendeley Plant Leaf Diseases Dataset with Augmentation](https://mendeley.com).

### Overview
* **Total Samples:** 10,000 images
* **Total Classes:** 10 distinct health states
* **Image Dimensions:** Resized to 128x128 pixels

### Class Distribution (1,000 images per class initially)
* `Tomato___Bacterial_spot`
* `Tomato___Early_blight`
* `Tomato___Late_blight`
* `Tomato___Leaf_Mold`
* `Tomato___Septoria_leaf_spot`
* `Tomato___Spider_mites Two-spotted_spider_mite`
* `Tomato___Target_Spot`
* `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
* `Tomato___Tomato_mosaic_virus`
* `Tomato___healthy`

### Visual Features Extracted by the CNN
* **Lesion Patterns:** Concentric rings, water-soaked spots, or irregular margins.
* **Color Variations:** Chlorosis (yellowing), necrosis (browning), or fungal mats.
* **Textural Disruption:** Wilting, puckering, or rust pustules.

---

## 🏗️ Model Architecture & Training
The network uses a sequential 9-layer CNN architecture designed to preserve spatial hierarchies while optimizing performance parameters.

```python
model = models.Sequential([
    layers.Input(shape=(128, 128, 3)),
    layers.Conv2D(32, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dropout(0.5),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.2),
    layers.Dense(num_classes, activation="softmax")
])
```

### Hyperparameters
* **Optimizer:** Adam
* **Loss Function:** `sparse_categorical_crossentropy`
* **Batch Size:** 32
* **Epochs:** 10

### Optimization History
* **Baseline Configuration:** Imbalanced raw classes caused severe overfitting.
* **Optimization Attempt 1:** Applied Early Stopping, but performance stalled between 35% and 40% accuracy.
* **Final Configuration:** Stabilized the dataset classes, dropped early stopping, and added strategically placed Dropout layers (0.5 and 0.2) to push final validation accuracy up to **90%**.

---

## 🔮 Limitations & Future Scope
* **Data Scale:** Expand the model past tomato plants to encompass diverse cash crops.
* **Infrastructure:** Deploy the backend microservices onto cloud infrastructure (AWS/GCP) for seamless integration with real-time field hardware.
* **Lifecycle Management:** Introduce automated MLOps tracking pipelines to monitor for data drift in changing environmental conditions.
