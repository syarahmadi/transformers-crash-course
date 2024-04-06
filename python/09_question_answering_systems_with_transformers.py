# -*- coding: utf-8 -*-
"""
# Question Answering Systems with Transformers using Python, PyTorch, and Huggingface

Question Answering (QA) systems are designed to answer questions posed in natural language. With the advent of transformer models, building powerful QA systems has become more accessible.

## 1. Introduction to Question Answering Systems

QA systems have been a significant area of research in NLP. The goal is to provide concise, relevant answers to user queries. Transformers, with their self-attention mechanism, have shown state-of-the-art performance in this domain.

## 2. Setting up the Environment

We'll be using Google Colab for this tutorial, which offers free GPU resources.
"""

# Install necessary packages
!pip install transformers
!pip install torch

"""After installation, let's verify the versions:

"""

import transformers
import torch

print("Transformers version:", transformers.__version__)
print("PyTorch version:", torch.__version__)

"""## 3. Loading Pre-trained Models

For QA tasks, the `BertForQuestionAnswering` model is commonly used.

"""

from transformers import BertTokenizer, BertForQuestionAnswering
from transformers import pipeline

# Load pre-trained model and tokenizer
model = BertForQuestionAnswering.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
tokenizer = BertTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

# QA pipeline
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

"""## 4. Using the QA System

With the model loaded, we can now pose questions based on a provided context.

"""

context = "Transformers are a type of deep learning model that have gained immense popularity in recent years due to their effectiveness in various NLP tasks."
question = "What have transformers gained in recent years?"

answer = qa_pipeline({'context': context, 'question': question})
print(answer)

"""## 5. Understanding the Output

The output will be a dictionary containing:

- `score`: Confidence score of the answer.
- `start`: Start position of the answer in the context.
- `end`: End position of the answer in the context.
- `answer`: The extracted answer from the context.

## 6. Conclusion

Question Answering systems have a wide range of applications, from chatbots to search engines. Transformers have revolutionized the performance and capabilities of these systems. This tutorial offers a glimpse into building a basic QA system. For advanced features and optimizations, the official documentation and related research papers are invaluable resources.

## 7. Reproducibility

Ensuring reproducibility involves:

1. Using the same package versions.
2. Setting random seeds for all libraries.
3. Using consistent model weights and architectures.

This tutorial serves as an introduction to QA systems using transformers. Dive deeper by exploring the official documentation, community discussions, and academic research on the topic.
"""
