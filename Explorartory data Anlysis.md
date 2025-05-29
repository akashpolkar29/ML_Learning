# Exploratory Data Analysis (EDA) in Machine Learning 🔍📊

## Introduction
Exploratory Data Analysis (EDA) is the process of analyzing and visualizing data to understand its patterns, trends, and relationships before applying machine learning models. It helps in detecting missing values, outliers, and distributions, making data-driven decisions easier.

## Key Steps in EDA

### 1. Importing Necessary Libraries
```python
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For advanced visualizations
```

### 2. Loading the Dataset
```python
df = pd.read_csv("data.csv")  # Load dataset
df.head()  # Display first 5 rows
```

### 3. Understanding the Data
#### Shape of the dataset (number of rows & columns)
```python
df.shape  
```

#### Basic summary
```python
df.info()  # Check data types and missing values
df.describe()  # Summary statistics (mean, std, min, max)
```

#### Checking for missing values
```python
df.isnull().sum()  # Count missing values per column
```

### 4. Data Cleaning
#### Handling Missing Values
```python
df.fillna(df.mean(), inplace=True)  # Replace missing values with column mean
```

#### Handling Duplicates
```python
df.drop_duplicates(inplace=True)  
```

### 5. Visualizing Data
#### Histogram (Distribution of a Column)
```python
df["age"].hist(bins=20)
plt.xlabel("Age")
plt.ylabel("Count")
plt.title("Age Distribution")
plt.show()
```

#### Box Plot (Detecting Outliers)
```python
sns.boxplot(x=df["salary"])
plt.title("Salary Outliers")
plt.show()
```

#### Correlation Heatmap (Feature Relationships)
```python
plt.figure(figsize=(8, 6))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Matrix")
plt.show()
```

#### Scatter Plot (Relation Between Two Features)
```python
sns.scatterplot(x=df["age"], y=df["salary"])
plt.title("Age vs Salary")
plt.show()
```

### 6. Detecting Outliers (Using IQR Method)
```python
Q1 = df["salary"].quantile(0.25)
Q3 = df["salary"].quantile(0.75)
IQR = Q3 - Q1
df = df[(df["salary"] >= (Q1 - 1.5 * IQR)) & (df["salary"] <= (Q3 + 1.5 * IQR))]
```

### 7. Feature Engineering
#### Encoding categorical variables
```python
df = pd.get_dummies(df, columns=["gender"], drop_first=True)  
```

#### Scaling numerical variables
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[["age", "salary"]] = scaler.fit_transform(df[["age", "salary"]])
```

