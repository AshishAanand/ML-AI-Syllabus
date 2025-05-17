# 📚 Machine Learning Complete Hierarchy

This document breaks Machine Learning into its **most detailed structure**, covering main categories, subcategories, and key algorithms. Use this as a comprehensive reference and roadmap.

---

## 🧠 1. Machine Learning Categories

- **Supervised Learning**
- **Unsupervised Learning**
- **Semi-Supervised Learning**
- **Reinforcement Learning**
- **Self-Supervised Learning**
- **Deep Learning**
- **ML Concepts**
- **Feature Engineering**
- **Tools & Libraries**

---

## ✅ 2. Supervised Learning
 - Learn from labeled data (inputs + outputs)

### ➤ Regression (predict continuous output)

- Linear Regression
- Polynomial Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Support Vector Regression (SVR)
- Decision Tree Regression
- Random Forest Regression
- Gradient Boosting Regression (GBR)
     - XGBoost
     - LightGBM
     - CatBoost

- Bayesian Regression
- K-Nearest Neighbors Regression (KNN Regr)

### ➤ Classification(predict discrete classes)

- Logistic Regression
- Naive Bayes
     - Gussian NB
     - Multinomial NB
     - Bernoulli NB

- K-Nearest Neighbors (KNN)
- Support Vector Machines (SVM)
- Decision Tree Classification
- Random Forest Classification
- Gradient Boosting Classification (GBC)
     - XGBoost
     - LightGBM
     - CatBoost

- Neural Networks
- Quadratic Discriminant Analysis (QDA)
- Linear Discriminant Analysis (LDA)
- Multi-Layer Perceptron (MLP)

---

## 🔍 3. Unsupervised Learning
  - Learn from unlabeled data (no outputs)

### ➤ Clustering
- K-Means
- Hierarchical Clustering
- DBSCAN
- Mean-Shift
- Gaussian Mixture Models (GMM)
- Agglomerative Clustering
- Spectral Clustering
- BIRCH

### ➤ Dimensionality Reduction
- Principal Component Analysis (PCA)
- Linear Discriminant Analysis (LDA) (also used in supervised)
- t-SNE (t-distributed Stochastic Neighbor Embedding)
- UMAP (Uniform Manifold Approximation and Projection)
- Factor Analysis
- Independent Component Analysis (ICA)
- Autoencoders

### ➤ Association Rule Learning
- Apriori
- Eclat
- FP-Growth

---

## 🧩 4. Semi-Supervised Learning
  - Mi of a small amount of labeled and a large amount of unlabeled data

Techniques:

- Self-training
- Label Propagation
- Transductive SVM (TSVM)
- Co-training
- Generative models(e.g., SSL GANs (Semi-Supervised GANs))

---

## 🎮 5. Reinforcement Learning
   - Learn through trial and error using rewards

### ➤ Value-Based Methods

- Q-Learning
- Deep Q-Networks (DQN)

### ➤ Policy-Based

- REINFORCE Algorithm
- Policy Gradient Methods

### ➤ Actor-Critic Methods

- A2C (Advantage Actor-Critic)
- A3C (Asynchronous Advantage Actor-Critic)
- DDPG (Deep Deterministic Policy Gradient)
- TD3 (Twin Delayed Deep Deterministic Policy Gradient)
- PPO (Proximal Policy Optimization)
- SAC (Soft Actor-Critic)

### ➤ Model-Based RL
   - Learn a model of the environment to plan actions

- Dyna-Q
- World Models

---

## 🧪 6. Self-Supervised Learning
   - Learn from structured data generated from unlabeled data

Applications:

- Contrastive Learning
  - SimCLR
  - MoCo
  - BYOL
- Masked Modeling
  - BERT
  - MAE (Masked Autoencoder)

---

## 🧬 7. Deep Learning ( Subfield of ML)

- Neural Networks
     - Percetrons
     - Deep Neural Networks (DNNs)
- Convolutional Neural Networks (CNNs)
     - For image processing
     - ResNet, VGG, Inception, EffcientNet
- Recurrent Neural Networks (RNNs)
     - For sequential Data
     - LSTM, GRU
- Transformer Models
     - Attention Mechanism
     - BERT
     - GPT
     - T5
     - Vision Transformers (ViT)
- Generative Models
     - GANs (DCGAN, CycleGAN, StyleGAN)
     - VAEs
- Autoencoders
     - Denoising Autoencoder
     - Variational Autoencoder (VAE)


---

##  🧠 Common Algorithm Families

### 1. Linear Models
- Linear Regression
- Logistic Regression
- Ridge Regression
- Lasso Regression
- ElasticNet
- Bayesian Linear Regression

### 2. Tree-Based Models
- Decision Trees
- Random Forest
- Extra Trees
- Gradient Boosting (GBM, XGBoost, LightGBM, CatBoost)
- Histogram-Based Gradient Boosting

### 3. Support Vector Machines
- SVM for Classification
- SVR (Support Vector Regression)
- Kernel SVM

### 4. Instance-Based Learning
- K-Nearest Neighbors (KNN)
- Radius Neighbors
- Locally Weighted Learning

### 5. Ensemble Methods
- Bagging
- Boosting
- Stacking
- Voting Classifier/Regressor

### 6. Probabilistic Models
- Naive Bayes (Gaussian, Multinomial, Bernoulli)
- Hidden Markov Models (HMM)
- Gaussian Mixture Models (GMM)
- Bayesian Networks

### 7. Neural Networks
- Perceptron
- Multi-Layer Perceptron (MLP)
- Convolutional Neural Networks (CNNs)
- Recurrent Neural Networks (RNNs)
- LSTM, GRU
- Transformers
- GANs (Generative Adversarial Networks)
- Autoencoders

---

## 📌 Key Concepts & Techniques

### 🔍 Model Evaluation Metrics

#### ➤ Classification
- Accuracy
- Precision
- Recall
- F1-Score
- ROC Curve & AUC
- Confusion Matrix
- Log Loss

#### ➤ Regression
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R² Score

---

### 🔁 Cross-Validation
- K-Fold Cross-Validation
- Stratified K-Fold
- Leave-One-Out (LOO)
- ShuffleSplit
- TimeSeriesSplit

---

### 🧠 Bias-Variance Tradeoff
- High Bias: Underfitting
- High Variance: Overfitting

---

### ⚙️ Regularization
- L1 Regularization (Lasso)
- L2 Regularization (Ridge)
- ElasticNet

---

### 🎛 Hyperparameter Tuning
- Grid Search
- Random Search
- Bayesian Optimization (Optuna, Hyperopt)
- Genetic Algorithms
- Cross-Validation inside tuning

---

### 🧪 Model Selection
- Baseline Models
- Comparing Models using CV
- Statistical Tests (t-test, ANOVA)
- Model Interpretability (SHAP, LIME)

---

### 🧬 Data Splitting
- Train / Validation / Test Split
- Stratified Split
- Holdout Validation

---

### 🧹 Data Cleaning
- Handling Missing Values (Imputation)
- Removing Duplicates
- Outlier Detection (IQR, Z-score, Isolation Forest)

---

### 🧮 Statistical Assumptions (for linear models)
- Linearity
- Independence
- Homoscedasticity
- Normality of Errors

---

## IV. 🛠️ Feature Engineering

### 1. 📏 Scaling / Normalization
- Min-Max Scaling
- Standardization (Z-score)
- Robust Scaling
- MaxAbs Scaling

### 2. 🧠 Encoding Categorical Features
- Label Encoding
- One-Hot Encoding
- Ordinal Encoding
- Frequency Encoding
- Target / Mean Encoding
- Binary Encoding
- Hash Encoding

### 3. 📊 Feature Selection Techniques

#### ➤ Filter Methods
- Correlation Thresholding
- Chi-Squared Test
- Mutual Information

#### ➤ Wrapper Methods
- Recursive Feature Elimination (RFE)
- Forward/Backward Feature Selection

#### ➤ Embedded Methods
- Lasso (L1)
- Tree Feature Importance

---

### 4. 🧱 Dimensionality Reduction
- PCA (Principal Component Analysis)
- t-SNE
- UMAP
- LDA (Linear Discriminant Analysis)
- Autoencoders

---

### 5. ⚗️ Feature Creation / Extraction
- Polynomial Features
- Binning (Discretization)
- Text Vectorization (TF-IDF, CountVectorizer, Word2Vec, BERT)
- Date/Time Features
- Domain-Specific Features (ratios, rates, etc.)
- Interaction Features

---

### 6. 🔧 Handling Missing Data
- Deletion (listwise/pairwise)
- Mean/Median/Mode Imputation
- KNN Imputation
- Iterative Imputer (MICE)
- Interpolation
- Indicator Variables

---

### 7. 📐 Outlier Handling
- Z-score
- IQR
- Isolation Forest
- DBSCAN (Density-based detection)

---

### 8. 🧹 Data Transformation
- Log Transformation
- Box-Cox / Yeo-Johnson
- Power Transformation

---

> ✅ This section is essential to master model performance, interpretability, and real-world deployment.

---

## 🧰 10. Libraries & Tools

- `scikit-learn` - ML models & utilities
- `TensorFlow` - Deep learning framework
- `PyTorch` - Deep learning framework
- `XGBoost` - Gradient boosting
- `Pandas` - Data manipulation
- `NumPy` - Numerical computing
- `OpenCV` - Image processing
- `Matplotlib / Seaborn` - Visualization

---

> ✅ Use this structure as your **study roadmap**, reference checklist, and skill-tracker while mastering Machine Learning and AI.

