# Unsupervised Learning Notes

## **1. What is Unsupervised Learning?**
Unsupervised learning is a type of machine learning where the model **learns patterns from data without labeled outputs**.
- Unlike supervised learning (where we have inputs and correct outputs), unsupervised learning **finds hidden structures** in data.

### 💡 **Example:**
Imagine you own a clothing store, and you have customer purchase data but don’t know their shopping behavior.
- Unsupervised learning can group customers into categories like **budget shoppers, mid-range buyers, and luxury customers** without any predefined labels.

### **Types of Unsupervised Learning:**
1. **Clustering** → Grouping similar data points together.  
   - Example: **K-Means Clustering** (finding customer segments).
2. **Dimensionality Reduction** → Reducing data complexity while keeping key information.  
   - Example: **PCA (Principal Component Analysis)** (compressing large datasets).

---

## **2. K-Means Clustering**
### **What is Clustering?**
Clustering is the process of grouping data points that are similar to each other.

### **How K-Means Works?**
1. Pick a number **K** (number of clusters).
2. Select **K random points** as cluster centers (centroids).
3. Assign each data point to the closest centroid.
4. Compute the new centroids by taking the **average** of all points in each cluster.
5. Repeat the process until centroids **don’t change much**.

### 💡 **Example:**
Imagine you have a basket of fruits, and you want to group them based on weight and sweetness.
- **Cluster 1:** Small & sour fruits (Lemon, Orange)
- **Cluster 2:** Medium & sweet fruits (Apples, Pears)
- **Cluster 3:** Large & very sweet fruits (Mangoes, Watermelons)

### **Elbow Method (Finding the Best K)**
The **Elbow Method** helps choose the best number of clusters.
- It plots the number of clusters vs. the sum of squared distances of points from their cluster centers.
- The best **K** is where the graph **bends like an elbow** (adding more clusters doesn’t improve results much).

---

## **3. PCA (Principal Component Analysis)**
### **Why do we need PCA?**
When working with large datasets with many features (columns), **some features might be redundant**. PCA helps by reducing dimensions while keeping essential information.

### **How PCA Works?**
1. Standardize the data (make sure all features have equal importance).
2. Compute the **covariance matrix** (to understand relationships between features).
3. Find **eigenvalues & eigenvectors** (which represent main components).
4. Select the **top principal components** (that explain most variance).
5. Transform the data into these new components, reducing complexity.

### 💡 **Example:**
Suppose we have student scores in **Math, Science, English, and History**. Instead of analyzing four subjects separately, PCA can reduce them to two components:
- **Component 1:** Science & Math
- **Component 2:** English & History

This allows us to see overall trends in student performance without unnecessary complexity.

---

## **4. Neural Networks Basics**
### **What is a Neural Network?**
A neural network is a model inspired by the **human brain**. It consists of layers of connected "neurons" that process information.

### **Basic Building Block: The Perceptron**
A **perceptron** is the simplest form of a neural network.
- It takes input features, **multiplies them by weights**, adds bias, and passes the result through an **activation function** to decide the output.

### 💡 **Example:**
Think of a perceptron like a **spam filter**:
- If an email contains "Free Money" → **Spam (1)**
- If it’s from a known contact → **Not Spam (0)**

### **Backpropagation - How Neural Networks Learn**
1. Make a prediction.
2. Calculate the **error** (difference from the actual output).
3. Adjust the **weights** to reduce the error using **Gradient Descent**.
4. Repeat until the model improves.

---

## **Summary **
✔ **Unsupervised Learning** → Finds patterns without labels.  
✔ **K-Means Clustering** → Groups similar data points (used for customer segmentation).  
✔ **PCA (Principal Component Analysis)** → Reduces data dimensions while keeping important features.  
✔ **Neural Networks** → Work like brain cells; **perceptron** is the basic unit.  
✔ **Backpropagation** → Adjusts weights to improve learning.  


