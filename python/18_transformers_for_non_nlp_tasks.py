# -*- coding: utf-8 -*-
"""
# Transformers for Non-NLP Tasks with PyTorch and Hugging Face

Transformers have revolutionized the field of natural language processing (NLP). However, their application extends beyond NLP to areas such as computer vision, time series analysis, and more. This tutorial explores how to use transformers for non-NLP tasks using PyTorch and Hugging Face's transformers library.

## Prerequisites
Before we begin, ensure you have a Google Colab notebook or a similar environment set up. We'll use Python 3.x, PyTorch, and Hugging Face's transformers library.

## Installation
Start by installing the necessary libraries. Run the following commands in your notebook:
"""

!pip install -q torch torchvision torchaudio
!pip install -q transformers

"""## Importing Libraries
After installation, import the required libraries:
"""

import torch
import transformers
from torch import nn
from torchvision import models, transforms
from transformers import ViTFeatureExtractor, ViTForImageClassification
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt

"""## Checking Package Versions
To ensure reproducibility, let's check the version of the installed packages:
"""

print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")

"""## Conceptual Overview
Transformers are models that leverage self-attention mechanisms to process input data in parallel, significantly improving efficiency and performance in various tasks. While originally designed for NLP, the adaptability of transformers allows them to excel in other domains, such as image classification (ViT - Vision Transformer) and time series forecasting.

## Vision Transformers (ViT)
Vision Transformers (ViT) apply transformer architecture to image classification tasks. Unlike conventional CNNs that process images through local convolutions, ViT divides an image into patches and processes these patches as a sequence, allowing the model to capture global dependencies.

## Example: Image Classification with ViT
Let's dive into an example of using a Vision Transformer for image classification.

### Step 1: Load and Prepare an Image
First, we need to load an image and prepare it for the model. We'll use the ViTFeatureExtractor for this purpose.##
"""

import torch
from torchvision import datasets, transforms
from PIL import Image
import matplotlib.pyplot as plt

# Define a transform to convert MNIST images to the format expected by ViT
transform = transforms.Compose([
    transforms.Resize(224),  # Resize the image to 224x224 pixels
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Repeat the single channel image 3 times to create a 3-channel image
    transforms.Normalize(0.5, 0.5),  # Normalize the image
])

# Load the MNIST dataset
mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Select an image from the dataset
image, label = mnist_dataset[0]  # Change 0 to any other index to select a different image

# Since ViTFeatureExtractor expects a PIL Image, we convert the tensor back to a PIL Image for demonstration
image_for_vit = transforms.ToPILImage(mode='RGB')(image)

# Display the image
plt.imshow(image_for_vit)
plt.title(f"MNIST Sample - Label: {label}")
plt.show()

"""### Step 2: Feature Extraction

The ViTFeatureExtractor processes the image into the format expected by the transformer model.
"""

feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')

inputs = feature_extractor(images=image_for_vit, return_tensors="pt")

"""### Step 3: Load the ViT Model

We load a pre-trained Vision Transformer model from Hugging Face's model hub.
"""

model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

"""### Step 4: Image Classification
Now, we can classify the image using the transformer model.
"""

outputs = model(**inputs)
logits = outputs.logits
predicted_class_idx = logits.argmax(-1).item()
print(f"Predicted class: {model.config.id2label[predicted_class_idx]}")

"""## Conclusion
In this tutorial, we've explored how transformers can be applied to non-NLP tasks, specifically image classification using Vision Transformers (ViT). The principles learned here can be extended to other domains such as time series analysis, where the sequential nature of transformers can be leveraged.

Remember, the field of AI and machine learning is rapidly evolving, and staying updated with the latest advancements is key to leveraging these powerful models effectively.
"""

