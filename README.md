# Auto MPG Regression with PyTorch — Full MLP Pipeline + Deep Insights

This repository implements a complete regression workflow using **PyTorch**, applied to the classic **Auto MPG** dataset.  
Rather than just providing code, this README explains **why each design choice matters**, including tensor shapes, architecture depth, optimizer logic, and metric computation.

It is written for learners who want to understand PyTorch deeply and build correct, production-grade training loops.

---

# 🧱 1. Project Overview

We build a model to predict **Miles Per Gallon (MPG)** using a fully connected neural network (MLP), achieving **R-square ~ 90%**. 

The dataset includes attributes such as:
- weight  
- displacement  
- horsepower  
- cylinders  
- acceleration  
- model year  

Model architecture:

```
Input → Linear → ReLU → Linear → ReLU → Linear → Output
```

Hidden sizes: 64 → 32  
Output size: 1

---

# ⚙️ 2. Run Instructions

Install dependencies:

```bash
pip install torch numpy pandas scikit-learn
```

Run the project:

```bash
python auto_mpg_mlp_regression.py
```

Ensure that the file:

```
Auto_MPG_data.csv
```

is placed in the same directory or adjust the `csv_path`.

---

# 📁 3. Project Structure

```
auto_mpg_mlp_regression.py
Auto_MPG_data.csv
README.md
```

---

# 🧪 4. Evaluation Metric — R² Score

We compute:

$ R^2 = 1 - \frac{SS_{res}}{SS_{tot}} $

Include:
- $SS_{res} = \sum (y_{true} - y_{pred})^2$ 
- $SS_{tot} = \sum (y_{true} - y_{mean})^2$ 

---


# 🧩 5. Key Theoretical Insights (Q1 → Q6)

This section answers every conceptual question tied to the training pipeline.

---

## ✅ (1) Why do we need:

```python
y_true = y_true.view(-1).float()
y_pred = y_pred.view(-1).float()
```

### ✔ 1. Avoid silent shape broadcasting bugs
`fc3` produces output `(B,1)`, while many parts of the pipeline use `(B,)`.

If shapes mismatch:

```
(B,1) - (B,)  →  PyTorch broadcasts to (B,B)
```

This results in:
- wrong SS_res  
- wrong SS_tot  
- wrong R²  
- NO error thrown  

Therefore we flatten tensors to 1D:

```
(B,1) → (B,)
```

### ✔ 2. Ensure floating-point math
If labels are accidentally `LongTensor`, subtraction and division behave incorrectly.  
`.float()` ensures numeric stability.

---

## ✅ (2) Difference between these two MLP definitions:

### Style A — two hidden layers

```python
self.fc1 = nn.Linear(input_dim, h1)
self.fc2 = nn.Linear(h1, h2)
self.fc3 = nn.Linear(h2, output_dim)
```

### Style B — one hidden layer

```python
self.linear1 = nn.Linear(input_dim, hidden_dim)
self.activation = nn.ReLU()
self.linear2 = nn.Linear(hidden_dim, output_dim)
```

### ✔ Insight:
Naming (`fc1` vs `linear1`) is **irrelevant**.

What matters is:
- Style A defines **two hidden layers**
- Style B defines **one hidden layer**

Two hidden layers allow deeper, more expressive models.

---

## ✅ (3) Why two hidden layers instead of one?

### ✔ 1. Function complexity
Auto MPG relationships are nonlinear:
- weight × horsepower interactions  
- displacement × year effects  
- cylinders × acceleration interactions  

One hidden layer *can* approximate any function but may require hundreds of neurons.  
Two hidden layers approximate complex functions with **fewer parameters** and more stability.

### ✔ 2. Hierarchical feature learning
Layer 1: learns primitive combinations  
Layer 2: learns higher-order combinations  

This mimics deep learning structure.

### ✔ 3. Educational purpose
Assignments often require two layers to illustrate:
- stacking nonlinearities  
- architecture depth  
- training dynamics  

---

## ✅ (4) Why use:

```python
x = x.squeeze(1)
```

### ✔ Reason:
`fc3` outputs `(B,1)` but computation of:

- MSE  
- R²  
- plotting  
- metric logging  
- loss curves  

all expect `(B,)`.

If not squeezed:
```
outputs shape = (B,1)
targets shape = (B,)
```

Then:

```
(B,1) - (B,)  → (B,B) broadcasting
```

This silently corrupts:
- loss  
- gradients  
- R²  

Fix: remove the redundant dimension:

```
(B,1) → (B,)
```

---

## ✅ (5) SGD vs Adam — which to choose?

### ✔ Why Adam is better here:
Adam uses *adaptive learning rates* and *momentum*:

### Why Adam is better here:

Adam uses _adaptive learning rates_ and _momentum_:

**Momentum estimate:**
$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
$

**Variance estimate:**
$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
$

**Bias correction:**
$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$

**Update rule:**
$
\theta_t = \theta_{t-1} - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$

### ✔ Intuition:
- If gradient direction is consistent → larger step  
- If gradient varies wildly → smaller step  
- Each parameter gets its own learning rate  
- Faster convergence than SGD  
- Less hyperparameter tuning  

### ✔ Why not SGD?
SGD requires:
- learning rate schedule  
- momentum tweaking  
- more training epochs  

For structured/tabular data like Auto MPG →  
👉 **Adam is superior and more stable.**

---

## ✅ (6) Why append predictions & targets *inside* the training loop?

### ✔ 1. R² requires all predictions across the epoch  
You cannot compute R² per batch.

### ✔ 2. `.detach()` prevents memory leak  
Without it, the entire autograd graph accumulates → GPU OOM.

### ✔ 3. `.cpu()` avoids Python list + CUDA issues  
Python lists store CPU tensors efficiently.

### ✔ 4. DataLoader yields batches  
You **must** collect inside the loop:

Correct:
```python
for X_batch, y_batch in train_loader:
    outputs = model(X_batch)
    all_preds.append(outputs.detach().cpu())
```

Incorrect / impossible:
```python
outputs = model(train_loader)
```

---

# 🧬 6. Full Pipeline Summary

1. RNG seed setup  
2. Load CSV  
3. Extract target `y` and features `X`  
4. Train/Val/Test split  
5. Standardize data  
6. Convert to PyTorch tensors  
7. Build `CustomDataset`  
8. Create DataLoaders  
9. Define MLP (2 hidden layers)  
10. Train model using Adam + MSE  
11. Track loss and R² each epoch  
12. Evaluate on validation  
13. Test model on final dataset  

---

# 🏁 7. Final Thoughts

This repository demonstrates:
- correct PyTorch training loop design  
- how to avoid silent broadcasting bugs  
- how Adam works internally  
- why deep MLPs outperform shallow ones  
- how to handle tensor shapes properly  
- best practices for metric computation  
- safe memory handling with `.detach()`  

By understanding these insights, you will write **far more stable, correct, and scalable PyTorch code** for real-world ML projects.

---

# 👤 Author

This README is designed to be:
- educational  
- technically accurate  
- professional  
- suitable for public GitHub repositories  
