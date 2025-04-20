# 🧠 Handwritten Digit Recognition with TensorFlow (MNIST Dataset)

In this project I used **TensorFlow** to build and train a basic Artificial Neural Network (ANN) that recognizes handwritten digits from the **MNIST dataset**. It also supports testing custom handwritten digits, such as your own drawings!

---

## 📦 Dataset: MNIST

- **60,000** training images
- **10,000** test images  
- Each image is **28x28 grayscale** representing digits `0-9`.

---

## 🛠️ Requirements

- Python 3.x
- TensorFlow
- NumPy
- Matplotlib
- scikit-learn
- OpenCV (`cv2`)
- PIL (Pillow)

---

## 🚀 How It Works

### 1. **Data Preprocessing**
```python
x_train, x_test = x_train / 255.0, x_test / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)
```
- Normalize pixel values to range [0, 1].
- One-hot encode labels.

---

### 2. **Model Architecture**
```python
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])
```

- A simple ANN with one hidden layer.
- Trained using `categorical_crossentropy` loss and `adam` optimizer.

---

### 3. **Training Results**
```text
Final Validation Accuracy ≈ 97.3%
Test Accuracy ≈ 97.4%
```

---

### 4. **Evaluate and Visualize Performance**

#### 📊 Confusion Matrix
- See how the model performs across each digit class.

#### ❌ Misclassified Digits
- Display examples the model got wrong.

---

## ✍️ Test on Your Own Handwriting

- Load your custom image (e.g., `"written_three_by_me.jpeg"`) 
- Preprocess:
  - Convert to grayscale
  - Resize to 28x28
  - Invert colors (`cv2.bitwise_not`)
  - Normalize
- Predict using the model:
```python
Predicted Digit: 3
```

---

## 🔍 Test on a Specific Sample from Test Set

Select a specific digit index (e.g., `124`) to visualize and test prediction.

---

## 🥳  Overall Result

Your model can now:
- Classify digits from the MNIST dataset with ~97% accuracy.
- Predict digits drawn by you if properly preprocessed.
---
## 👩‍💻 Author

**Yassmine Yazidi**  
Business Student & Data Enthusiast  
Tunisia 🇹🇳
