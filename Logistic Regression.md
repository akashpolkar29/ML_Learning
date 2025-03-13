### **Logistic Regression (Sigmoid Function, Decision Boundary)**

Logistic Regression is a **classification algorithm** used to predict the probability that a given input point belongs to a particular class. It is widely used for **binary classification** problems, where the output is a class label like "0" or "1," "spam" or "not spam," etc.

### **Key Concepts of Logistic Regression**

1. **Sigmoid Function:**
   - The core of logistic regression is the **Sigmoid function**, which converts the output of the model (a linear combination of inputs) into a probability between 0 and 1.
   - The sigmoid function is defined as:

     ```math
     \sigma(z) = \frac{1}{1 + e^{-z}}
     ```

     Where:
     - \(z = w_1 x_1 + w_2 x_2 + \dots + w_n x_n + b\) is the linear combination of input features and weights.
     - \(e\) is the base of the natural logarithm.

   - The output of the sigmoid function (\( \sigma(z) \)) is interpreted as the probability of the instance belonging to the positive class (usually labeled as "1").
   
2. **Decision Boundary:**
   - The **decision boundary** is the threshold that helps the model decide between two classes (e.g., 0 or 1).
   - In logistic regression, we typically use a threshold of **0.5**:
     - If \( \sigma(z) \geq 0.5 \), predict class "1".
     - If \( \sigma(z) < 0.5 \), predict class "0".
   - This decision boundary is a straight line (or hyperplane in higher dimensions), and the model classifies points on one side of the boundary as one class, and points on the other side as the other class.

---

### **Steps to Apply Logistic Regression**

Let’s break down the steps to apply logistic regression with a simple example.

---

### **Example Problem:**
Suppose we have a dataset of students, and we want to predict whether they pass (1) or fail (0) based on the number of hours they studied.

| Hours Studied | Passed (1/0) |
|---------------|--------------|
| 1             | 0            |
| 2             | 0            |
| 3             | 0            |
| 4             | 1            |
| 5             | 1            |
| 6             | 1            |

We’ll predict whether a student passes or fails based on the number of hours studied.

---

### **Step 1: Import Libraries**

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

---

### **Step 2: Prepare the Dataset**

```python
# Dataset: Hours studied and whether passed (1) or failed (0)
data = {'Hours': [1, 2, 3, 4, 5, 6],
        'Passed': [0, 0, 0, 1, 1, 1]}

df = pd.DataFrame(data)
print(df)
```

---

### **Step 3: Split the Data into Features and Target**

```python
X = df[['Hours']]
y = df['Passed']
```

---

### **Step 4: Split the Data into Training and Testing Sets**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

---

### **Step 5: Create and Train the Logistic Regression Model**

```python
model = LogisticRegression()
model.fit(X_train, y_train)
```

---

### **Step 6: Make Predictions**

```python
y_pred = model.predict(X_test)
```

---

### **Step 7: Evaluate the Model**

```python
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100}%")
```

---

### **Step 8: Visualize the Sigmoid Function and Decision Boundary**

```python
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Testing Data')

x_values = np.linspace(0, 7, 100)
y_values = model.predict_proba(x_values.reshape(-1, 1))[:, 1]

plt.plot(x_values, y_values, color='red', label='Decision Boundary (Sigmoid)')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression: Pass or Fail Prediction')
plt.legend()
plt.show()
```

---

### **Step 9: Predict New Values (Optional)**

```python
new_data = np.array([[3.5]])
predicted_probability = model.predict_proba(new_data)[:, 1]
print(f"Predicted probability of passing for 3.5 hours of study: {predicted_probability[0]:.2f}")
```

---

### **Conclusion**

You’ve applied **Logistic Regression** to predict whether a student will pass or fail based on the hours they studied. The steps involved:

1. **Prepare the dataset**: Separate the features (hours) and target (pass/fail).
2. **Split the data**: Divide the dataset into training and testing sets.
3. **Train the model**: Use the `LogisticRegression` algorithm.
4. **Make predictions**: Predict the pass/fail status.
5. **Evaluate the model**: Calculate accuracy.
6. **Visualize**: Plot the data points and decision boundary.
7. **Predict new values**: Make predictions for new data.

---

### **Summary of Key Concepts**

- **Sigmoid Function**: Converts the linear combination of inputs into probabilities (between 0 and 1).
- **Decision Boundary**: The threshold that separates the two classes. It is determined by the sigmoid function output.
- **Logistic Regression**: A classification algorithm used to model the probability of a binary outcome based on one or more input features.

By following these steps, you can apply logistic regression to classify binary outcomes in a variety of problems.
