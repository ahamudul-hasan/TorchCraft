# Building an Artificial Neural Network (ANN) with PyTorch

This folder demonstrates building a complete **multi-class image classification pipeline** from scratch using PyTorch. The model is trained on a subset of the **Fashion MNIST** dataset to classify clothing items into 10 categories.

---

## What is an ANN?

An **Artificial Neural Network (ANN)** is a computational model inspired by the human brain. It consists of layers of interconnected nodes (neurons):

- **Input Layer** — receives raw feature data
- **Hidden Layers** — learn abstract representations through transformations
- **Output Layer** — produces final predictions (class scores)

Each layer applies:

$$
z = X \cdot W + b
$$

followed by a non-linear **activation function** (like ReLU) to allow the network to learn complex patterns.

---

## Dataset: Fashion MNIST

The `fmnist_small.csv` file contains a subset of the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset.

| Property   | Value                                                                |
| ---------- | -------------------------------------------------------------------- |
| Image size | 28 × 28 pixels (784 features)                                       |
| Classes    | 10 clothing categories                                               |
| Format     | CSV — first column is label, remaining 784 columns are pixel values |

### The 10 Classes

| Label | Item        |
| ----- | ----------- |
| 0     | T-shirt/top |
| 1     | Trouser     |
| 2     | Pullover    |
| 3     | Dress       |
| 4     | Coat        |
| 5     | Sandal      |
| 6     | Shirt       |
| 7     | Sneaker     |
| 8     | Bag         |
| 9     | Ankle boot  |

---

## Pipeline Overview

```
CSV File
   ↓
Load with Pandas
   ↓
Split features (X) and labels (y)
   ↓
Train / Test Split (80/20)
   ↓
Normalize pixel values (÷ 255)
   ↓
CustomDataset → DataLoader (batches of 32)
   ↓
Define ANN Architecture (3 layers)
   ↓
Training Loop (100 epochs, SGD, CrossEntropyLoss)
   ↓
Evaluate on Test Set (accuracy)
```

---

## Step-by-Step Explanation

### Step 1: Imports

```python
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
```

All the core libraries are imported:

- `pandas` — for loading and inspecting the CSV data
- `sklearn` — for splitting data into train/test sets
- `torch` — the core PyTorch tensor library
- `Dataset`, `DataLoader` — for efficient batch loading
- `nn` — for building the neural network layers
- `optim` — for the optimizer (SGD)
- `matplotlib` — for visualizing sample images

---

### Step 2: Reproducibility

```python
torch.manual_seed(42)
```

Setting a manual seed ensures that random operations (weight initialization, data shuffling) produce the same results every time the notebook is run. This makes experiments reproducible.

---

### Step 3: Load and Explore Data

```python
df = pd.read_csv('fmnist_small.csv')
df.head()
```

The CSV is loaded into a Pandas DataFrame. Each row is one image:

- Column `0` → class label (0–9)
- Columns `1` to `784` → pixel intensity values (0–255)

A 4×4 grid of the first 16 images is plotted using Matplotlib to visually verify the data loaded correctly.

---

### Step 4: Separate Features and Labels

```python
x = df.iloc[:, 1:].values   # all pixel columns → shape (N, 784)
y = df.iloc[:, 0].values    # label column      → shape (N,)
```

- `x` contains the pixel data (the input features)
- `y` contains the integer class labels (the targets)

---

### Step 5: Train / Test Split

```python
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```

- **80%** of the data → training set
- **20%** of the data → test set
- `random_state=42` ensures the split is the same every run

---

### Step 6: Feature Scaling (Normalization)

```python
x_train = x_train / 255.0
x_test  = x_test  / 255.0
```

Pixel values originally range from **0 to 255**. Dividing by 255 rescales them to the range **[0, 1]**.

**Why normalize?**

- Prevents large input values from causing unstable gradients
- Speeds up convergence — the optimizer doesn't have to deal with very large or very small numbers
- Keeps inputs in a consistent range across all features

> **Important:** only fit the scaling on training data. The same divisor (255) is applied to test data to prevent data leakage.

---

### Step 7: Custom Dataset Class

```python
class CustomDataset(Dataset):

    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels   = torch.tensor(labels,   dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]
```

A custom class that wraps the NumPy arrays in PyTorch tensors.

| Method          | Purpose                                                                                                           |
| --------------- | ----------------------------------------------------------------------------------------------------------------- |
| `__init__`    | Converts NumPy arrays to tensors. Features use`float32`, labels use `long` (required by `CrossEntropyLoss`) |
| `__len__`     | Returns the number of samples so PyTorch knows the dataset size                                                   |
| `__getitem__` | Returns a single (feature, label) pair at index`idx` — used by DataLoader internally                           |

---

### Step 8: DataLoaders

```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)
```

| Parameter         | Value         | Reason                                                                       |
| ----------------- | ------------- | ---------------------------------------------------------------------------- |
| `batch_size`    | 32            | Process 32 images at a time — memory efficient                              |
| `shuffle=True`  | training only | Randomize order each epoch to prevent the model memorizing sequence patterns |
| `shuffle=False` | test only     | Order doesn't matter for evaluation                                          |

Each iteration over a loader yields a batch of shape `(32, 784)` for features and `(32,)` for labels.

---

### Step 9: Neural Network Architecture

```python
class MyNN(nn.Module):

    def __init__(self, num_features):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )

    def forward(self, x):
        return self.model(x)
```

#### Architecture Diagram

```
Input (784)
    ↓
Linear(784 → 128)
    ↓
ReLU
    ↓
Linear(128 → 64)
    ↓
ReLU
    ↓
Linear(64 → 10)
    ↓
Output (10 class scores / logits)
```

#### Layer Breakdown

| Layer                   | Input Size | Output Size | Purpose                                          |
| ----------------------- | ---------- | ----------- | ------------------------------------------------ |
| `nn.Linear(784, 128)` | 784        | 128         | First hidden layer, reduces dimensionality       |
| `nn.ReLU()`           | —         | —          | Adds non-linearity, removes negative activations |
| `nn.Linear(128, 64)`  | 128        | 64          | Second hidden layer, further abstraction         |
| `nn.ReLU()`           | —         | —          | Adds non-linearity again                         |
| `nn.Linear(64, 10)`   | 64         | 10          | Output layer, one score per class                |

#### What is ReLU?

**ReLU (Rectified Linear Unit)** is defined as:

$$
\text{ReLU}(z) = \max(0, z)
$$

- Outputs zero for any negative input
- Passes positive inputs unchanged
- Prevents the network from collapsing into a purely linear model
- Computationally cheap and avoids the vanishing gradient problem that plagues `sigmoid` in deep networks

#### Why `nn.Sequential`?

`nn.Sequential` stacks layers so that each layer's output is automatically passed as the next layer's input. This keeps the architecture clean and readable.

#### Why No Softmax in the Output?

The output layer produces raw **logits** (unnormalized scores). A softmax is **not applied** because `nn.CrossEntropyLoss` internally combines `LogSoftmax + NLLLoss`, so applying softmax manually would be redundant and slightly less numerically stable.

---

### Step 10: Hyperparameters, Loss & Optimizer

```python
epochs        = 100
learning_rate = 0.1

model     = MyNN(x_train.shape[1])       # num_features = 784
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)
```

#### Hyperparameters

| Parameter         | Value | Purpose                                      |
| ----------------- | ----- | -------------------------------------------- |
| `epochs`        | 100   | Number of full passes over the training data |
| `learning_rate` | 0.1   | Step size for each weight update             |

#### Loss Function: `CrossEntropyLoss`

Used for **multi-class classification**. It computes:

$$
\mathcal{L} = -\sum_{c=1}^{C} y_c \log(\hat{p}_c)
$$

where $y_c$ is the true class indicator and $\hat{p}_c$ is the predicted probability for class $c$.

- Penalizes confident wrong predictions heavily
- Internally applies softmax to the raw logits before computing the log loss

#### Optimizer: SGD (Stochastic Gradient Descent)

Updates parameters using:

$$
\theta \leftarrow \theta - \eta \cdot \nabla_\theta \mathcal{L}
$$

SGD updates weights after each batch (mini-batch gradient descent), which:

- Is faster than full-batch gradient descent
- Introduces some noise that can help escape local minima

---

### Step 11: Training Loop

```python
for epoch in range(epochs):

    total_epoch_loss = 0

    for batch_features, batch_labels in train_loader:

        outputs = model(batch_features)          # forward pass
        loss    = criterion(outputs, batch_labels)  # compute loss

        optimizer.zero_grad()                    # clear old gradients
        loss.backward()                          # backpropagation
        optimizer.step()                         # update weights

        total_epoch_loss += loss.item()

    avg_loss = total_epoch_loss / len(train_loader)
    print(f'Epoch: {epoch + 1} , Loss: {avg_loss}')
```

#### What Happens Each Epoch

1. **Forward pass** — features flow through the network layer by layer to produce output logits
2. **Loss computation** — CrossEntropyLoss compares predictions to true labels
3. **Zero gradients** — old gradients from the previous step are cleared (`zero_grad()` must be called before `backward()` to avoid gradient accumulation)
4. **Backward pass** — PyTorch's autograd computes ∂Loss/∂weight for every parameter using the chain rule
5. **Weight update** — SGD adjusts each weight by a small step in the direction that reduces loss
6. **Avg loss logged** — total batch losses are averaged and printed per epoch to monitor convergence

---

### Step 12: Evaluation

```python
model.eval()

total   = 0
correct = 0

with torch.no_grad():
    for batch_features, batch_labels in test_loader:
        outputs = model(batch_features)
        _, predicted = torch.max(outputs, 1)
        total   += batch_labels.shape[0]
        correct += (predicted == batch_labels).sum().item()

print(correct / total)
```

#### Key Points

| Code                      | Meaning                                                                                                                             |
| ------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| `model.eval()`          | Switches model to evaluation mode — disables dropout and batch norm training behavior (not strictly needed here but good practice) |
| `torch.no_grad()`       | Disables gradient computation during inference — saves memory and speeds up the forward pass                                       |
| `torch.max(outputs, 1)` | Returns the index of the highest logit per row — this index is the predicted class                                                 |
| `correct / total`       | Final**accuracy** on the test set                                                                                             |

The `_` (underscore) in `_, predicted = torch.max(outputs, 1)` discards the actual maximum value — only the predicted class index matters.

---

## Summary of the Full Pipeline

| Stage            | Tools Used                                                   |
| ---------------- | ------------------------------------------------------------ |
| Data loading     | `pandas.read_csv`                                          |
| Train/test split | `sklearn train_test_split`                                 |
| Normalization    | Manual division by 255                                       |
| Dataset wrapping | `torch.utils.data.Dataset`                                 |
| Batch loading    | `torch.utils.data.DataLoader`                              |
| Model definition | `nn.Module`, `nn.Sequential`, `nn.Linear`, `nn.ReLU` |
| Loss function    | `nn.CrossEntropyLoss`                                      |
| Optimizer        | `torch.optim.SGD`                                          |
| Backpropagation  | `loss.backward()` + `optimizer.step()`                   |
| Evaluation       | `torch.no_grad()`, `torch.max()`                         |

---

## Key Concepts Recap

| Concept                    | Explanation                                                                  |
| -------------------------- | ---------------------------------------------------------------------------- |
| **Forward Pass**     | Data flows input → hidden layers → output, producing predictions           |
| **Loss**             | A number measuring how wrong the predictions are                             |
| **Backpropagation**  | Calculates how much each weight contributed to the loss using the chain rule |
| **Gradient Descent** | Moves each weight slightly in the direction that reduces loss                |
| **Epoch**            | One full pass through the entire training dataset                            |
| **Batch**            | A small subset of the training data processed at one time                    |
| **Logits**           | Raw, unnormalized output scores before softmax                               |
| **Accuracy**         | Fraction of test samples the model predicted correctly                       |
