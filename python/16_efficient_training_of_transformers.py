# -*- coding: utf-8 -*-
"""
# Efficient Transformer Training with PyTorch and Hugging Face

## Introduction

Transformer models have reshaped Natural Language Processing (NLP) and are increasingly used for computer vision tasks. However, their massive size and computational demands pose challenges for efficient training. In this tutorial, we'll explore strategies for optimizing Transformer training using PyTorch and Hugging Face while ensuring our code runs smoothly in Google Colab.

Here we use the following **efficiency techniques**:

* Mixed Precision Training: Automatic mixed precision (FP16) accelerates computations and reduces memory usage without heavily compromising accuracy.
* Gradient Scaling: Prevents numerical issues (underflow/overflow) commonly arising in mixed-:precision training.
* GPU Utilization: The code assumes GPU availability (device, torch.cuda.amp), significantly boosting performance for Transformer training.

## Environment Setup

Let's start by setting up a Colab environment and installing packages:
"""

!pip install -q transformers torch matplotlib datasets

"""Import necessary libraries and check versions:"""

import transformers
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset

print(f"Transformers version: {transformers.__version__}")
print(f"Torch version: {torch.__version__}")

"""## Dataset Preparation: Text Classification Example"""

dataset = load_dataset('glue', 'sst2')  # Let's use the SST-2 sentiment classification task

# Preprocessing
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize(batch):
  return tokenizer(batch['sentence'], padding='max_length', truncation=True, max_length=512)

dataset = dataset.map(tokenize, batched=True)

"""## Model Definition"""

model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

"""## Optimization Techniques"""

# **1. Optimizer: AdamW**
# * Employs the AdamW optimizer, a sophisticated optimization algorithm with weight decay for regularization.
# * Gathers all trainable weights and biases from the Transformer model using `model.parameters()`.
# * Uses a learning rate of 0.00005 (specified by `lr=5e-5`) to control the pace of weight updates.
optimizer = AdamW(model.parameters(), lr=5e-5)

# **2. Training and Validation Split**
# * Calculates the number of samples to include in the training set (90% of the 'train' subset of your dataset).
# Training and validation splits
train_size = int(0.9 * len(dataset['train']))
train_dataset, val_dataset = dataset['train'].train_test_split(train_size).values()

# **3. Custom Collate Function (`collate_fn`)**
# * Defines how to assemble a batch of individual samples, handling variable-length sequences in Transformer models.
def collate_fn(batch):
    """Custom collate function to create tensors and transfer to device"""
    input_ids = torch.tensor([example['input_ids'] for example in batch])
    attention_mask = torch.tensor([example['attention_mask'] for example in batch])
    labels = torch.tensor([example['label'] for example in batch])

    return {
        'input_ids': input_ids.to(device),
        'attention_mask': attention_mask.to(device),
        'labels': labels.to(device)
    }

# **4. Data Loaders**
# * Creates data loaders for efficient batching during training and validation.
# * Shuffles training data (`shuffle=True`) for better generalization.
# * Employs the `collate_fn` to prepare batches.
from torch.utils.data import DataLoader
train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=4, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=4, collate_fn=collate_fn)


# **5. Learning Rate Scheduler**
# * Implements a scheduler to decrease the learning rate linearly over time for stable training.
# * Configures the scheduler based on the total number of training steps.
num_epochs = 1
num_training_steps = num_epochs * len(train_dataloader)
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0, # Default value
                                            num_training_steps=num_training_steps)

print(dataset['train'][0])

"""## Training Loop with Efficiency

Here, we utilize mixed precision.

### Key Benefits of Mixed Precision

* Speed: FP16 computations are generally much faster on modern GPUs, leading to significant training acceleration.
* Memory: Using FP16 reduces memory footprint, allowing you to train larger models or use larger batch sizes.

### Important Considerations

* Hardware: Mixed-precision benefits are most pronounced on GPUs with Tensor Cores (NVIDIA architecture).
* Stability: Not all models are equally suited for FP16. If you encounter NaN (Not a Number) values or instability, adjust your code or consider a less aggressive mixed-precision approach.

In the following part of the code:


```
scaler = torch.cuda.amp.GradScaler()
```

We instantiate a GradScaler object from PyTorch's automatic mixed precision (AMP) package. The GradScaler is crucial for preventing underflow or overflow issues often arising when working with mixed precision.

Then we use:

```
with torch.cuda.amp.autocast():
```

This context manager enables automatic mixed-precision within its block. Here's how it works:
* Casting: It automatically converts model operations and inputs to half-precision (FP16), where possible, for computational speed gains.
* Gradient Scaling: During backward propagation, gradients are scaled up to prevent vanishing gradients that can occur in FP16 computations.

```
outputs = model(**batch)
```

The forward pass of your Transformer model is executed within the autocast() context. Suitable computations will benefit from the efficiency of FP16.

```
loss = outputs.loss
```

The loss calculation is likely performed in full-precision (FP32) for numerical stability.

To integrate the scaling during the backward pass:

```
scaler.scale(loss).backward()  
scaler.step(optimizer)
scaler.update()
```
"""

from tqdm import tqdm  # Progress visualization

# WARNING: Turning this to true may take more GPU!
evaluate_in_loop = False

def evaluate(dataloader):
    model.eval()
    total_loss, total_correct = 0.0, 0

    with tqdm(dataloader, unit="batch") as teval:  # Progress visualization
        for batch in teval:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.item() * batch['input_ids'].size(0)
            predictions = outputs.logits.argmax(dim=-1)
            total_correct += (predictions == batch['labels']).sum().item()

            # Update progress bar description
            teval.set_description(f"Evaluating: Loss {loss.item():.3f}")

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = total_correct / len(dataloader.dataset)

    return {'loss': avg_loss, 'accuracy': accuracy}

# Training loop
for epoch in range(num_epochs):
    model.train()
    for batch in tqdm(train_dataloader):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss

        # Mixed precision
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    if evaluate_in_loop:
      # Evaluation
      results = evaluate(val_dataloader)
      print(f"Epoch {epoch+1}, Validation Loss: {results['loss']:.3f}, Validation Accuracy: {results['accuracy']:.3f}")

"""## Evaluation"""

model.eval()
val_acc = evaluate(val_dataloader)
print(f"Validation Accuracy: {val_acc}")

