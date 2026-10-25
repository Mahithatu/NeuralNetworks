# Fashion-MNIST Classification with PyTorch

A machine learning project that implements and evaluates a feed-forward neural network (Multi-Layer Perceptron) for image classification using the Fashion-MNIST dataset.

The primary objective of this project is to understand how neural network architecture and training hyperparameters influence model performance. Various hidden-layer configurations, learning rates, activation functions, dropout rates, and training durations were evaluated to identify an effective model.

---

## Project Objectives

This notebook explores:

- Building a fully connected neural network from scratch in PyTorch
- Classifying Fashion-MNIST images into 10 clothing categories
- Experimenting with hidden-layer configurations
- Evaluating the impact of learning rates
- Analyzing the effect of training epochs on performance
- Understanding the role of activation functions and dropout
- Evaluating model performance using multiple classification metrics

---

## Dataset

The project uses the **Fashion-MNIST** dataset, a collection of grayscale images of clothing items published by Zalando Research.

### Dataset Characteristics

- 70,000 images
  - 60,000 training images
  - 10,000 testing images
- 28 × 28 grayscale pixels
- 10 classes

| Label | Category |
|---------|----------|
| 0 | T-Shirt/Top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle Boot |

---

## Model Architecture

The notebook implements a Multi-Layer Perceptron (MLP) using PyTorch.

### Final Architecture

```text
Input Layer (784)
        ↓
Linear(784 → 256)
        ↓
ReLU
        ↓
Linear(256 → 128)
        ↓
ReLU
        ↓
Linear(128 → 64)
        ↓
ReLU
        ↓
Linear(64 → 10)
        ↓
Output Layer (10 Classes)
```

ReLU activation was applied after each hidden layer to introduce non-linearity.

No activation function was applied to the output layer because PyTorch's `CrossEntropyLoss` internally applies `LogSoftmax` during training.

---

## Experiments

Several experiments were conducted to understand the relationship between model complexity and predictive performance.

### Hidden Layer Experiments

The input layer contains 784 neurons and the output layer contains 10 neurons.

Using the Two-Thirds Rule as an initial guideline, the total number of hidden neurons was kept below approximately 533 neurons.

The initial architecture consisted of two hidden layers with 256 and 128 neurons:

```text
784 → 256 → 128 → 10
```

This configuration achieved an accuracy of **46.96%**.

To increase model capacity, a third hidden layer was added:

```text
784 → 256 → 128 → 64 → 10
```

This resulted in a significant improvement, increasing accuracy to **76.36%**.

A larger network was also evaluated:

```text
784 → 512 → 128 → 32 → 10
```

However, this architecture failed to converge effectively and achieved only **10% accuracy**.

### Dropout Experiment

Dropout regularization was introduced to evaluate its impact on model performance.

A dropout rate of **0.2** was applied after the first hidden layer in the architecture:

```text
784 → 256 → 128 → 64 → 10
```

The resulting accuracy decreased to **60.35%**.

Since the baseline model did not exhibit significant overfitting, dropout reduced performance rather than improving generalization and was therefore removed from the final model.

### ReLU Experiment

To evaluate the importance of activation functions, ReLU was removed from the final two hidden layers.

This caused model accuracy to decrease dramatically to **28%**, demonstrating the importance of non-linear activation functions in deep neural networks.

All hidden-layer experiments were conducted using a learning rate of **0.01**.

---

### Learning Rate Experiments

The initial learning rate was set to **0.01**.

For the architecture:

```text
784 → 256 → 128 → 64 → 10
```

the model achieved an accuracy of **76.36%** after 100 epochs.

Additional learning rates were evaluated:

| Learning Rate | Accuracy |
|--------------|----------|
| 0.10 | 10.00% |
| 0.05 | 10.00% |
| 0.01 | 76.36% |
| 0.001 | 84.68% |

Lowering the learning rate to **0.001** significantly improved convergence and classification performance.

---

### Epoch Experiments

For the architecture:

```text
784 → 256 → 128 → 64 → 10
```

with a learning rate of **0.01**:

| Epochs | Accuracy |
|----------|----------|
| 100 | 76.36% |
| 150 | 80.10% |
| 200 | 81.14% |

With a learning rate of **0.001**:

| Epochs | Accuracy |
|----------|----------|
| 100 | 84.68% |
| 150 | 85.61% |
| 200 | 86.28% |

Increasing the number of training epochs consistently improved model performance, although the rate of improvement diminished as training progressed.

---

## Final Training Configuration

### Loss Function

```python
nn.CrossEntropyLoss()
```

CrossEntropyLoss was selected because Fashion-MNIST is a multi-class classification problem.

### Optimizer

```python
torch.optim.Adam()
```

Adam optimizer was chosen for its adaptive learning rate capabilities and efficient convergence behavior.

### Final Hyperparameters

| Parameter | Value |
|------------|---------|
| Architecture | 784-256-128-64-10 |
| Hidden Layers | 3 |
| Activation Function | ReLU |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Learning Rate | 0.001 |
| Epochs | 200 |

---

## Evaluation Metrics

Model performance was evaluated using multiple classification metrics rather than relying solely on accuracy.

The following metrics were calculated:

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

Metrics were evaluated both at the overall dataset level and at the individual class level to better understand model behavior and classification performance across all Fashion-MNIST categories.

---

## Best Performing Model

| Parameter | Value |
|------------|---------|
| Architecture | 784-256-128-64-10 |
| Learning Rate | 0.001 |
| Epochs | 200 |
| Accuracy | 86.28% |
| Precision | 86.41% |
| Recall | 86.28% |
| F1 Score | 86.26% |
| Training Time | ~4 minutes |

---

## Results

| Hidden Layers | Dropout | Learning Rate | Epochs | Accuracy | Precision | Recall | F1 Score | Training Time |
|--------------|----------|----------|----------|----------|----------|----------|----------|----------|
| 784-256-128-64-10 | None | 0.001 | 100 | 0.8468 | 0.8455 | 0.8468 | 0.8451 | ~2 min |
| 784-256-128-64-10 | None | 0.01 | 100 | 0.7636 | 0.7532 | 0.7636 | 0.7488 | ~2 min |
| 784-256-128-64-10 | None | 0.05 | 100 | 0.1000 | 0.0100 | 0.1000 | 0.0181 | ~2 min |
| 784-256-128-64-10 | None | 0.10 | 100 | 0.1000 | 0.0100 | 0.1000 | 0.0181 | ~2 min |
| 784-256-128-64-10 | 0.2 | 0.01 | 100 | 0.6035 | 0.5803 | 0.6035 | 0.5658 | ~2 min |
| 784-256-128-64-10 | None | 0.01 | 150 | 0.8010 | 0.7954 | 0.8010 | 0.7957 | ~3 min |
| 784-256-128-64-10 | None | 0.01 | 200 | 0.8114 | 0.8073 | 0.8114 | 0.8077 | ~4 min |
| 784-256-128-64-10 | None | 0.001 | 150 | 0.8561 | 0.8582 | 0.8561 | 0.8551 | ~3 min |
| 784-256-128-64-10 | None | 0.001 | 200 | 0.8628 | 0.8641 | 0.8628 | 0.8626 | ~4 min |
| 784-512-128-32-10 | None | 0.01 | 100 | 0.1000 | 0.0100 | 0.1000 | 0.0181 | ~2 min |
| 784-256-128-10 | None | 0.01 | 100 | 0.4695 | 0.4440 | 0.4695 | 0.4049 | ~2 min |

### Key Observations

- Network architecture significantly influences model performance.
- ReLU activations are critical for effective learning.
- Learning rate selection has a major impact on convergence and final accuracy.
- Additional training epochs improve performance, although gains diminish over time.
- Dropout regularization is not always beneficial when overfitting is not present.
- Comprehensive evaluation using Precision, Recall, F1 Score, and Confusion Matrix provides deeper insights than accuracy alone.

---

## Technologies Used

- Python
- PyTorch
- NumPy
- Matplotlib
- Scikit-Learn
- Jupyter Notebook

---

## Repository Structure

```text
.
├── Fashion_MNIST.ipynb
└── README.md
```

---

## Key Takeaways

This project provided hands-on experience with:

- Neural network implementation in PyTorch
- Hyperparameter tuning
- Multi-class image classification
- Model evaluation using advanced metrics
- Understanding the impact of architecture design on performance
- Practical experimentation with activation functions, dropout, learning rates, and training duration

The best-performing model achieved **86.28% accuracy** on the Fashion-MNIST test dataset using a simple feed-forward neural network architecture.