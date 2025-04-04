## Confusion Matrix in Machine Learning

A Confusion Matrix is a table used to evaluate the performance of a classification model. It compares the actual labels with the predicted labels, helping to understand the model’s accuracy and errors.

### Structure of a Confusion Matrix
For a binary classification (e.g., spam vs. not spam), the confusion matrix looks like this:

| Actual / Predicted | Predicted Positive (1) | Predicted Negative (0) |
|--------------------|-----------------------|-----------------------|
| Actual Positive (1) | True Positive (TP) ✅ | False Negative (FN) ❌ |
| Actual Negative (0) | False Positive (FP) ❌ | True Negative (TN) ✅ |

- **True Positive (TP):** Correctly predicted positive cases.
- **False Positive (FP):** Incorrectly predicted as positive (Type I Error).
- **False Negative (FN):** Incorrectly predicted as negative (Type II Error).
- **True Negative (TN):** Correctly predicted negative cases.

### Example
Imagine a model that detects if an email is spam (1) or not spam (0). If we test it on 100 emails, the confusion matrix might look like:

| Actual / Predicted | Spam (1) | Not Spam (0) |
|--------------------|---------|-------------|
| Spam (1) | 40 (TP) | 10 (FN) |
| Not Spam (0) | 5 (FP) | 45 (TN) |

### Key Metrics Derived from Confusion Matrix
#### Accuracy 📊:
\[
Accuracy = \frac{TP + TN}{TP + TN + FP + FN}
\]
Measures overall correctness.

#### Precision 🎯 (Positive Predictive Value):
\[
Precision = \frac{TP}{TP + FP}
\]
Out of all predicted positives, how many are actually correct?

#### Recall (Sensitivity or True Positive Rate):
\[
Recall = \frac{TP}{TP + FN}
\]
Out of all actual positives, how many were correctly predicted?

#### F1 Score (Balance between Precision & Recall):
\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

### Confusion Matrix in Python
You can create a confusion matrix using Scikit-Learn:
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Example actual vs predicted values
y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]
y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 0]

# Compute confusion matrix
cm = confusion_matrix(y_true, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Spam", "Spam"], yticklabels=["Not Spam", "Spam"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```
