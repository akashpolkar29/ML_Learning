# Linear Regression 📈🔢

Linear Regression is a supervised learning algorithm used to model the relationship between independent variables (features `X`) and dependent variables (label `Y`) by fitting a straight line.

## Mathematical Form of Linear Regression

\[
Y = WX + b
\]

Where:
- \( Y \) = Predicted output (label)
- \( X \) = Input features
- \( W \) = Weights (coefficients)
- \( b \) = Bias (intercept)

---

## 1️⃣ Ordinary Least Squares (OLS) Method 📏

OLS is a method to find the best-fitting line by minimizing the sum of squared errors (residuals).

### Error (Residual) for Each Data Point

\[
\text{Error} = Y_{\text{actual}} - Y_{\text{predicted}}
\]

### Cost Function (Mean Squared Error - MSE)

\[
J(W, b) = \frac{1}{n} \sum_{i=1}^{n} (Y_i - (WX_i + b))^2
\]

The goal is to find \( W \) and \( b \) that minimize this function.

### Python Implementation of OLS

```python
from sklearn.linear_model import LinearRegression  

# Training Data
X = [[1], [2], [3], [4], [5]]  # Feature (independent variable)
y = [2, 4, 6, 8, 10]           # Label (dependent variable)

# Model Initialization & Training
model = LinearRegression()
model.fit(X, y)

# Predictions
pred = model.predict([[6]])  # Predict for X=6
print(f"Prediction for X=6: {pred[0]:.2f}")  # Output: 12.00
