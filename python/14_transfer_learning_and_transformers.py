# -*- coding: utf-8 -*-
"""
#Fine-Tuning Transformers with PyTorch and Hugging Face

## Introduction
This tutorial demonstrates how to fine-tune a pre-trained transformer model for a text classification task using PyTorch and Hugging Face's transformers library. We'll focus on sentiment analysis, modifying a pre-trained model to better understand the sentiment of text. This process is vital for adapting general models to specific needs or datasets.

##Setting Up the Environment
First, ensure you're using a Google Colab notebook with GPU support for efficient training.

##Install Necessary Packages
"""

!pip install -q torch torchvision
!pip install -q transformers
!pip install -q matplotlib

"""## Import Libraries

"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AdamW
import matplotlib.pyplot as plt

"""## Understanding Transformers and Transfer Learning
Transformers use self-attention mechanisms to capture relationships in data, excelling in tasks requiring context understanding. Transfer learning adapts a pre-trained model to a new, but related, task, saving on resources and training time.

## Loading a Pre-Trained Model
We'll start with a model pre-trained on a large corpus and fine-tune it for our specific task.
"""

model_name = "distilbert-base-uncased"
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""## Preparing the Dataset
Assume you have a dataset with texts and corresponding sentiment labels. Here's how to preprocess it:
"""

# Sample data
texts = ["I love this product!", "Worst experience ever."]
labels = [1, 0]  # 1 for positive, 0 for negative sentiment

# Tokenize and encode texts
inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
labels = torch.tensor(labels)

# Create a torch dataset
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
loader = DataLoader(dataset, batch_size=2)

"""## Fine-Tuning the Model
Adjust the pre-trained model to your specific task by continuing the training process.

## Define Optimizer and Training Loop
"""

optimizer = AdamW(model.parameters(), lr=1e-5)

model.train()
for epoch in range(3):  # number of epochs
    for batch in loader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch}, Loss: {loss.item()}")

"""## Save the Fine-Tuned Model
After fine-tuning, save the model for later use.
"""

model.save_pretrained("my_finetuned_model")

"""## Making Predictions
With the fine-tuned model, you can now make predictions on new data.
"""

def predict_sentiment(text):
    model.eval()
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        logits = model(**inputs).logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)
    return probabilities

"""## Visualizing Predictions
Visualize the sentiment predictions as a bar chart.
"""

def show_prediction(text):
    probabilities = predict_sentiment(text)
    plt.bar(range(len(probabilities[0])), probabilities[0])
    plt.title('Sentiment Prediction')
    plt.xticks(range(len(probabilities[0])), ['Negative', 'Positive'])
    plt.ylabel('Probability')
    plt.show()

# Test with a new text
show_prediction("I love using transformers for NLP!")

"""## Ensuring Reproducibility
Note the versions of the key packages used for others to reproduce your results.
"""

import transformers
print("Torch Version:", torch.__version__)
print("Transformers Version:", transformers.__version__)

"""## Conclusion
Fine-tuning pre-trained transformers allows you to leverage their power for your specific tasks and datasets. It's a potent way to achieve high performance with less data and computational resources. Stay updated with the latest models and practices, and always monitor your model's performance.
"""

