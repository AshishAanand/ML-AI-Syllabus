# 🚀 Machine Learning Algorithms: When & Why to Use Them

---

## 🔀 Supervised Learning

### 🔹 1. **Linear Regression**

* **Use when**: Predicting a continuous value with a *linear relationship*.
* **Examples**: House price prediction, salary estimation
* **Assumes**: Features have a linear correlation with the target
* **Don’t use when**: Data has nonlinear patterns or outliers dominate

---

### 🔹 2. **Polynomial Regression**

* **Use when**: Relationship is nonlinear but follows a polynomial curve
* **Examples**: Predicting population growth, performance curves
* **Don’t use when**: Too high a degree → risk of overfitting

---

### 🔹 3. **Ridge / Lasso / ElasticNet Regression**

* **Use when**: You want to **avoid overfitting** in linear models
* **Ridge**: Best for multicollinearity (L2 penalty)
* **Lasso**: Best for feature selection (L1 penalty)
* **ElasticNet**: Combines both (L1 + L2)

---

### 🔹 4. **Logistic Regression**

* **Use when**: Target is **binary** (0 or 1, yes/no)
* **Examples**: Spam detection, cancer prediction (yes/no)
* **Don’t use when**: Target is not categorical or decision boundary is not linear

---

### 🔹 5. **K-Nearest Neighbors (KNN)**

* **Use when**: Simple classification or regression with small datasets
* **Examples**: Recommendation systems, image classification
* **Works well when**: Data has no noise, low dimensions
* **Don’t use when**: Large dataset or high dimensionality

---

### 🔹 6. **Decision Trees**

* **Use when**: You want an interpretable model
* **Examples**: Loan approval, disease diagnosis
* **Pros**: Easy to visualize, handles both regression/classification
* **Cons**: Prone to overfitting (use pruning or ensembles)

---

### 🔹 7. **Random Forest**

* **Use when**: You want a powerful, accurate model
* **Examples**: Stock prediction, medical diagnosis
* **Works well when**: Many features and you want to reduce overfitting
* **Don’t use when**: Need for real-time predictions (can be slow)

---

### 🔹 8. **Support Vector Machine (SVM)**

* **Use when**: You have clear margin of separation in data
* **Examples**: Face detection, bioinformatics
* **Best for**: Medium-sized datasets with clear decision boundaries
* **Don’t use when**: Dataset is too large or noisy

---

### 🔹 9. **Gradient Boosting / XGBoost / LightGBM**

* **Use when**: You want **state-of-the-art performance**
* **Examples**: Kaggle competitions, fraud detection, credit scoring
* **Pros**: Handles missing data, nonlinearities, ranking problems
* **Cons**: More complex and harder to interpret

---

## 🧩 Unsupervised Learning

### 🔸 1. **K-Means Clustering**

* **Use when**: You want to group similar data points
* **Examples**: Customer segmentation, market research
* **Works well when**: Clusters are spherical and balanced
* **Don’t use when**: Clusters vary in shape/size

---

### 🔸 2. **Hierarchical Clustering**

* **Use when**: You want to visualize clusters in a dendrogram
* **Examples**: Document classification, taxonomy
* **Best for**: Small to medium-sized datasets

---

### 🔸 3. **DBSCAN**

* **Use when**: Data has noise and irregular cluster shapes
* **Examples**: Geospatial data, anomaly detection
* **Don’t use when**: Data is high-dimensional and sparse

---

## ⚠️ Overfitting vs Underfitting

| Term         | Meaning                                      | Solution                                    |
| ------------ | -------------------------------------------- | ------------------------------------------- |
| Overfitting  | High training accuracy, low test accuracy    | Regularization, more data, simplification   |
| Underfitting | Poor performance on both train and test data | Use more complex model, feature engineering |

---

## 📊 Metric Guide

| Task Type     | Use These Metrics                        |
| ------------- | ---------------------------------------- |
| Regression    | MAE, MSE, RMSE, R²                       |
| Binary Class. | Accuracy, Precision, Recall, F1, ROC-AUC |
| Multiclass    | Confusion Matrix, Macro F1, Top-k Acc    |
| Clustering    | Silhouette Score, Davies-Bouldin Index   |

---

## 🧠 Feature Scaling?

| Algorithm Type      | Requires Scaling? |
| ------------------- | ----------------- |
| Linear/Logistic Reg | ✅ Yes             |
| KNN, SVM            | ✅ Yes             |
| Tree-based models   | ❌ No              |
| Naive Bayes         | ❌ No              |

---

## 🧠 Regularization Techniques

| Technique      | Use Case                         |
| -------------- | -------------------------------- |
| **Ridge**      | Many features, multicollinearity |
| **Lasso**      | Feature selection and sparsity   |
| **ElasticNet** | Combine Ridge + Lasso benefits   |

---

## 🏁 Workflow (TL;DR)

```
1. Define Problem → 
2. Collect + Clean Data → 
3. Preprocess + Validate → 
4. Train (CV + Scaling) → 
5. Evaluate (Metrics) → 
6. Tune Hyperparameters → 
7. Deploy + Monitor
```

---
