# -*- coding: utf-8 -*-
"""
# Customizing Transformer Architectures in PyTorch and Hugging Face
## Introduction
Transformers have become the backbone of modern NLP tasks, offering significant improvements in understanding context and relationships in text. Customizing transformer architectures allows researchers and practitioners to tailor models to specific needs, enhancing performance and efficiency for particular tasks. This advanced tutorial will guide you through customizing transformer architectures using PyTorch and Hugging Face's transformers library.

## Setting Up Your Environment
To begin, ensure you're using a Google Colab notebook with GPU support for efficient model training and manipulation.

## Install Necessary Packages
"""

!pip install -q torch torchvision
!pip install -q transformers
!pip install -q matplotlib

"""## Import Libraries"""

import torch
from torch import nn
from transformers import BertModel, BertConfig, AutoTokenizer
import matplotlib.pyplot as plt

"""## Understanding Transformer Architecture
A transformer model comprises several key components: embeddings, self-attention mechanisms, and feed-forward neural networks. Customizing these elements can lead to significant improvements or adaptations for specific tasks.

## Customizing Components
### 1. Custom Embeddings
Embeddings transform input tokens into vectors of a specified dimension. Customizing embeddings can adapt how the model interprets input data.

Example: Adding Positional Encoding
"""

class CustomEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        # Ensure the maximum positions align with BERT's expected maximum
        self.max_position_embeddings = config.max_position_embeddings  # typically 512
        self.position_embeddings = nn.Embedding(self.max_position_embeddings, config.hidden_size)

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        # Ensure position_ids do not exceed the maximum expected by the model
        if seq_length > self.max_position_embeddings:
            raise ValueError(f"Input sequence length ({seq_length}) exceeds maximum allowed length ({self.max_position_embeddings}).")

        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        embeddings = word_embeddings + position_embeddings
        return embeddings

"""### 2. Custom Self-Attention
Self-attention is a mechanism that allows the model to weigh the importance of different parts of the input data.

Example: Custom Attention Head
"""

class CustomAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Custom attention logic here
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer

"""### 3. Custom Feed-Forward Networks
Feed-forward networks process the outputs from the attention mechanism. Customizing these can allow for more complex interactions between the model's learned representations.

Example: Custom Feed-Forward Layer
"""

class CustomFeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        self.intermediate_act_fn = nn.GELU()  # or any other activation function
        self.output_dense = nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, attention_output):
        intermediate_output = self.dense(attention_output)
        intermediate_output = self.intermediate_act_fn(intermediate_output)
        layer_output = self.output_dense(intermediate_output)
        return layer_output

"""## Assembling the Custom Transformer
Now, let's put together our custom components into a transformer model.
"""

class CustomBertModel(nn.Module):
    def __init__(self, config):
        super(CustomBertModel, self).__init__()
        # Load pre-trained BERT model
        self.bert = BertModel(config)

        # Binary classification layer: Maps from hidden size to 1 output
        self.classifier = nn.Linear(config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        # Pass inputs through BERT model
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Get the last hidden states
        last_hidden_state = outputs.last_hidden_state  # Shape: [batch_size, sequence_length, hidden_size]

        # Apply mean pooling to get a fixed size output (aggregate across all tokens)
        pooled_output = torch.mean(last_hidden_state, dim=1)  # Shape: [batch_size, hidden_size]

        # Pass pooled output through classifier to get the final logit
        logit = self.classifier(pooled_output)  # Shape: [batch_size, 1]

        return logit.squeeze(-1)  # Shape: [batch_size]

"""## Dataloader

We create a synthetic dataset for our experiments.
"""

import torch
from torch.utils.data import DataLoader, TensorDataset

# Define the correct maximum input ID based on BERT's vocabulary and maximum sequence length
max_vocab_size = 30522  # BERT's vocabulary size
max_seq_length = 512  # Maximum sequence length for BERT

num_samples = 10  # Number of samples in your dataset

# Create random data representing your input_ids for the model
# Ensure all input IDs are within the vocabulary size range
input_ids = torch.randint(low=0, high=max_vocab_size, size=(num_samples, max_seq_length))

# Assuming a binary classification task, create random labels
labels = torch.randint(low=0, high=2, size=(num_samples,))

# Create a TensorDataset and DataLoader
dataset = TensorDataset(input_ids, labels)
data_loader = DataLoader(dataset, batch_size=8, shuffle=True)

"""# Training the Custom Model
To train your custom model, you'll need a dataset, a loss function, and an optimizer. Here's a simplified training loop:
"""

# Assume you have a DataLoader `data_loader` with input_ids and labels
config = BertConfig.from_pretrained('bert-base-uncased')
model = CustomBertModel(config)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

# Define the loss function
loss_function = torch.nn.BCEWithLogitsLoss()

model.train()
for epoch in range(3):  # number of epochs
  # In your training loop
  for input_ids, labels in data_loader:
      optimizer.zero_grad()
      outputs = model(input_ids)
      outputs = outputs.squeeze(-1)  # Remove the last dimension to match label shape
      loss = loss_function(outputs, labels.float())
      loss.backward()
      optimizer.step()

"""## Conclusion
Customizing transformers allows you to tailor models to the specific nuances of your task, potentially leading to better performance and more efficient training. Experiment with different configurations and components to find what works best for your specific needs.

## Versioning for Reproducibility
Ensure reproducibility by noting the version of the key libraries used:
"""

import transformers
print("Torch Version:", torch.__version__)
print("Transformers Version:", transformers.__version__)

