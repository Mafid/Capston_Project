# Multimodal Measles Classification

This repository contains the complete implementation of a **Multimodal Convolutional Neural Network (CNN)** designed to diagnose measles by fusing raw clinical image data with binary patient metadata (e.g., Fever, Cough, Red Eyes, Koplik Spots, and Vaccination Status). This hybrid approach bridges visual deep learning with structured clinical indicators to achieve robust classification.

---

## Table of Contents
1. [Architecture Overview](#1-architecture-overview)
2. [System Requirements & Installation](#2-system-requirements--installation)
3. [Data Pipeline & Preprocessing](#3-data-pipeline--preprocessing)
4. [Multimodal Model Design](#4-multimodal-model-design)
5. [Training & Validation](#5-training--validation)
6. [Evaluation Metrics & Diagnostics](#6-evaluation-metrics--diagnostics)
7. [Inference Pipeline (Clinical Simulation)](#7-inference-pipeline--clinical-simulation)

---

## 1. Architecture Overview

The system utilizes a late-fusion dual-branch network architecture:
* **Image Branch (Vision):** Leverages a pre-trained **MobileNetV2** (frozen initially for feature extraction) taking standard `(224, 224, 3)` clinical images.
* **Metadata Branch (Structured Clinical Features):** Passes a 5-dimensional binary clinical observation vector through a series of Fully Connected (Dense) layers with Dropout regularization.
* **Fusion & Classifier:** Concatenates both latent feature representations, feeding them into a dense fusion network culminating in a single-unit Sigmoid classifier predicting the probability of measles.

```
[Image Input (224,224,3)] -> [MobileNetV2 Base] -> [GAP] -> [Dense 128 (ReLU)] ---
                                                                                |--> [Concatenate] -> [Dense 64 (ReLU)] -> [Sigmoid Output]
[Metadata Input (5-dim)]  -> [Dense 32 (ReLU)]   ---------> [Dense 16 (ReLU)] ---
```

---

## 2. System Requirements & Installation

Ensure you have Python 3.8+ installed along with a GPU-enabled environment (such as Google Colab or local CUDA-enabled environments).

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib seaborn pillow roboflow
```

---

## 3. Data Pipeline & Preprocessing

### 3.1 Metadata Loading & Cleaning
* Standardize metadata feature headings and target outputs.
* Convert patient clinical parameters into float-based binary values (`0` or `1`).
* Missing patient attributes are filled using zero imputation.

```python
# Clean column names and map labels to binary targets
df.columns = df.columns.str.strip()
df["label"] = df["Label"].astype(str).str.lower().str.strip()
label_map = {"normal": 0, "measles": 1}
df["target"] = df["label"].map(label_map)
```

### 3.2 Automated Image Retrieval & Alignment
This pipeline retrieves images from a Roboflow project using a COCO-annotation format and performs regex normalization to map image filenames in the CSV metadata dataset to local files:

```python
# Normalize filenames to perform precise matching
def normalize_name(name):
    name = name.lower()
    name = re.split(r"\.rf\.", name)[0]
    name = re.sub(r"(_jpg|_jpeg|_png|\.jpg|\.jpeg|\.png)$", "", name)
    name = re.sub(r"[^a-z0-9]", "", name)
    return name
```

### 3.3 TensorFlow Data Pipeline (`tf.data` API)
To optimize performance, images are parsed asynchronously and mapped alongside clinical metadata:

```python
def load_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, [224, 224])
    return tf.cast(image, tf.float32) / 255.0
```

---

## 4. Multimodal Model Design

The neural network is compiled using TensorFlow's Functional API to accommodate multiple input streams:

```python
# Image Feature Extraction
image_input = layers.Input(shape=(224, 224, 3), name="image_input")
base_model = MobileNetV2(include_top=False, weights="imagenet", input_tensor=image_input)
base_model.trainable = False
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)

# Metadata Feature Extraction
metadata_input = layers.Input(shape=(5,), name="metadata_input")
m = layers.Dense(32, activation="relu")(metadata_input)
m = layers.Dropout(0.2)(m)
m = layers.Dense(16, activation="relu")(m)

# Latent Space Fusion
combined = layers.Concatenate()([x, m])
combined = layers.Dense(64, activation="relu")(combined)
combined = layers.Dropout(0.3)(combined)
output = layers.Dense(1, activation="sigmoid", name="output")(combined)

model = Model(inputs=[image_input, metadata_input], outputs=output)
```

---

## 5. Training & Validation

The network utilizes **Binary Cross-Entropy Loss** and is optimized with the **Adam** optimizer. Regularization is actively managed using dynamic callbacks:
* **Early Stopping:** Prevents overfitting by monitoring validation loss with a patience threshold of `7` epochs.
* **Learning Rate Plateau Reduction:** Automatically scales down the learning rate by a factor of `0.5` if validation loss stagnates for `3` epochs.

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)
```

---

## 6. Evaluation Metrics & Diagnostics

After training, predictions are verified using scikit-learn metrics to measure clinical precision and sensitivity:
* **Accuracy:** Overall correctness score.
* **Precision:** Ratio of true positive diagnoses to total predicted positive diagnoses (critical for reducing false alarms).
* **Recall (Sensitivity):** Ability of the model to identify all positive measles cases (vital in disease containment).
* **F1-Score:** Harmonic mean of precision and recall.

Performance metrics are visualized using a classification report and a confusion matrix heatmap.

---

## 7. Inference Pipeline (Clinical Simulation)

Deploy the model for point-of-care environments by processing a newly uploaded patient image and gathering diagnostic observations in real-time:

```python
# Feed the image and clinical metadata vector directly into the model
probability = model.predict({
    "image_input": preprocessed_image,
    "metadata_input": patient_metadata
})[0][0]
```

### Real-time Diagnostic Visualization
The system automatically prints a final diagnostic verdict (Measles vs. Normal) with a confidence percentage, rendered alongside the clinical image for medical review.
```
