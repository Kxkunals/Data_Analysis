# 📊 Linear Regression – Custom Implementations (OLS, BGD, L1 Regularized)

This project contains three custom implementations of Linear Regression:

1. **OLS (Normal Equation)**
2. **Batch Gradient Descent (BGD)**
3. **L1-Regularized BGD (Lasso)**

Each class follows the same structure: `fit`, `predict`, `returnScore`, and `getEquation`.

---

## 1️⃣ **Ordinary Least Squares (OLS)**

### 🔬 Algorithm:

Solves the linear regression analytically using the **Normal Equation**:

$$
\theta = (X^T X)^{-1} X^T y
$$

### 📌 Steps:

1. Add bias term to feature matrix:

   $$
   X \leftarrow [\mathbf{1}, X]
   $$

2. Apply normal equation:

   $$
   \theta = (X^T X)^{-1} X^T y
   $$

3. Predict:

   $$
   \hat{y} = X \cdot \theta
   $$

4. Score (R²):

   $$
   R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
   $$

---

## 2️⃣ **Batch Gradient Descent (BGD)**

### 🔁 Algorithm:

Instead of directly solving the equation, this version uses **BGD** to iteratively minimize the MSE:

$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \hat{y}^{(i)})^2
$$

Update rule per iteration:

$$
\theta := \theta - \alpha \cdot \nabla_\theta
$$

Where:

$$
\nabla_\theta = -\frac{2}{n} X^T(y - X\theta)
$$

### 🔄 Steps:

1. Add bias:

   $$
   X \leftarrow [\mathbf{1}, X]
   $$

2. Initialize weights to zeros:

   $$
   \theta = [0, 0, ..., 0]
   $$

3. For each epoch:

   * Compute predictions: $\hat{y} = X\theta$
   * Compute error: $e = y - \hat{y}$
   * Compute gradient:

     $$
     \nabla_\theta = -\frac{2}{n} X^T e
     $$
   * Update:

     $$
     \theta := \theta - \alpha \cdot \nabla_\theta
     $$

4. Predict and evaluate just like OLS.

---

## 3️⃣ **L1 Regularized BGD (Lasso)**

### 🔐 Algorithm:

Applies **L1 regularization** to induce sparsity in the coefficients.

Loss function with regularization:

$$
J(\theta) = \frac{1}{n} \sum_{i=1}^{n} (y^{(i)} - \hat{y}^{(i)})^2 + \lambda \sum_{j=1}^{m} |\theta_j|
$$

Update rule:

$$
\theta := \theta - \alpha \left( \nabla_\theta + \lambda \cdot \text{sign}(\theta) \right)
$$

### 🧾 Steps:

1. Add bias:

   $$
   X \leftarrow [\mathbf{1}, X]
   $$

2. Initialize weights to zero.

3. For each epoch:

   * Compute error: $e = y - X\theta$
   * Compute gradient:

     $$
     \nabla_\theta = -\frac{2}{n} X^T e + \lambda \cdot \text{sign}(\theta)
     $$
   * Update:

     $$
     \theta := \theta - \alpha \cdot \nabla_\theta
     $$

4. Final model has **sparse weights** (some set to 0).

---

## 🧠 Common Functions

### ✅ `predict(X_test)`

Adds bias and returns predictions:

$$
\hat{y} = X \cdot \theta
$$

---

### 📈 `returnScore(X_test, y_test)`

Computes the R² score:

$$
R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
$$

---

### ✍️ `getEquation()`

Returns a string of the model:

$$
\hat{y} = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n
$$

---

