# -*- coding: utf-8 -*-
"""
# Working with HuggingFace's Transformers Library

In this tutorial, we'll explore the Transformers library by HuggingFace, a powerful tool for working with state-of-the-art NLP models like BERT, GPT-2, and more.

## Prerequisites:
Basic understanding of PyTorch.
Familiarity with deep learning concepts.

## Setting up the Environment
First, let's set up our Colab environment:
"""

!pip install -q torch torchvision transformers

"""## 1. Loading Pre-trained Models
HuggingFace's Transformers library provides a plethora of pre-trained models. Let's start by loading the BERT model.
"""

from transformers import BertTokenizer, BertModel

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

"""## 2. Tokenization
Before feeding text to BERT, we need to tokenize it. The tokenizer will convert our text into tokens that correspond to BERT's vocabulary.
"""

text = "Hello, HuggingFace!"
encoded_input = tokenizer(text, return_tensors='pt')
print(encoded_input)

"""## 3. Model Inference
Now, let's use the BERT model to get embeddings for our text.
"""

import torch
with torch.no_grad():
    output = model(**encoded_input)

# Extract the sequence output (representations for each token in the input)
sequence_output = output.last_hidden_state
print(sequence_output)

"""## 4. Using Transformer Models for Tasks
HuggingFace provides interfaces for specific tasks like sequence classification, token classification, etc. Let's use BERT for sequence classification.
"""

from transformers import BertForSequenceClassification

# Load the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Example input text
texts = ["Hello, HuggingFace!", "Transformers are awesome!"]

# Tokenize and get predictions
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
outputs = model(**encoded_inputs)

# Get the logits from the model's output
logits = outputs.logits
print(logits)

"""## 5. Fine-tuning on Custom Data
For this example, let's assume we have a binary classification task. We'll create dummy data and fine-tune BERT on it.
"""

# Dummy data
texts = ["I love Transformers.", "I don't like this library."]
labels = torch.tensor([1, 0])  # 1: Positive, 0: Negative

# Tokenize the data
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')
encoded_inputs["labels"] = labels

# Fine-tuning
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
model.train()

for epoch in range(3):
    optimizer.zero_grad()
    outputs = model(**encoded_inputs)
    loss = outputs.loss
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

"""## 6. Saving & Loading Fine-tuned Models
After fine-tuning, you might want to save your model for later use.
"""

# Save model
model.save_pretrained("./my_bert_model")

# Load model
loaded_model = BertForSequenceClassification.from_pretrained("./my_bert_model")

