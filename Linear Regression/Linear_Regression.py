import torch
import torch.nn as nn

# Simulate one-hot encoded labels for a batch of 3 examples and 2 classes
one_hot_labels = torch.tensor([[0, 1], [1, 0], [0, 1]])  # Shape: [3, 2]

# Convert one-hot labels to class indices
labels = torch.argmax(one_hot_labels, dim=1)  # Shape: [3], now each element is a class index

# Simulate model output (logits) for a batch of 3 examples, 2 classes
outputs = torch.randn(3, 2)  # Shape: [3, 2]

print(one_hot_labels)

print(labels)

print(outputs)