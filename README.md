# Isolation Forest Outlier Detection

This project demonstrates how to use the **Isolation Forest** algorithm from Scikit-learn to detect **outliers and anomalies** in a dataset.

The project creates synthetic data containing two normal clusters and then adds randomly generated points as potential outliers. An Isolation Forest model is trained on the normal data and used to identify unusual observations.

## 📌 Project Overview

**Isolation Forest** is an unsupervised machine learning algorithm designed for anomaly detection.

The main idea is that **outliers are easier to isolate than normal data points**. The algorithm builds random decision trees and identifies observations that can be separated from the rest of the data with fewer splits.

## 🛠️ Technologies Used

* Python
* NumPy
* Scikit-learn
* Isolation Forest

## 📊 Dataset

The dataset is generated using NumPy rather than using an external dataset.

Normal data is generated using:

```python
X = 0.3 * np.random.randn(100, 2)
```

Two normal clusters are then created:

```python
X_train = np.r_[X + 2, X - 2]
```

The test dataset contains the two normal clusters plus 20 randomly generated points:

```python
X_test = np.r_[X + 2, X - 2,
               np.random.uniform(low=-6, high=6, size=(20, 2))]
```

This gives:

* **200 normal training points**
* **220 test points**
* **20 randomly generated potential outliers**

## 🤖 Model

The project uses:

```python
IsolationForest(contamination=0.1, random_state=42)
```

### Parameters

* `contamination=0.1` → assumes approximately 10% of the data may be anomalous.
* `random_state=42` → makes the results reproducible.

## 🔍 Predictions

The trained model predicts whether each test observation is normal or anomalous.

The Isolation Forest prediction values are:

```text
1  → Normal point
-1 → Anomaly / Outlier
```

The predictions are generated using:

```python
predictions = model.predict(X_test)
```

## 📁 Project Structure

```text
Isolation-Forest/
│
├── main.py
└── README.md
```

## 🚀 How to Run

### 1. Clone the repository

```bash
git clone https://github.com/Vedline2547/Isolation-forest.git
```

### 2. Navigate into the project

```bash
cd Isolation-Forest
```

### 3. Install the required libraries

```bash
pip install numpy scikit-learn
```

### 4. Run the Python script

```bash
python isolation_forest.py
```

## 📈 Example Output

The program prints predictions similar to:

```text
Predictions: [ 1  1  1  1 ... -1 -1  1 -1 ...]
```

Where:

* `1` represents a normal observation.
* `-1` represents an observation identified as an anomaly.

Because the data is randomly generated, the exact predictions may vary between runs.

## 🎯 Learning Objectives

Through this project, I learned how to:

* Generate synthetic data using NumPy.
* Create datasets containing normal observations and potential outliers.
* Understand the concept of anomaly detection.
* Implement Isolation Forest using Scikit-learn.
* Train an unsupervised anomaly detection model.
* Interpret Isolation Forest predictions.
* Distinguish between normal observations and anomalies.

## 📚 Key Concept

Isolation Forest works on the principle that **anomalies are rare and different from normal observations**, making them easier to isolate using random decision trees.

This makes Isolation Forest useful for applications such as:

* Fraud detection
* Network intrusion detection
* Fault detection
* Data quality monitoring
* Unusual transaction detection
* Industrial anomaly detection

## 👨‍💻 Author

**Vedline Ochieng**

Civil Engineering Student | Machine Learning & AI Enthusiast | Python Developer
