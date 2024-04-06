# -*- coding: utf-8 -*-
"""

#Comprehensive Fine-tuning of Transformers for Text Classification using PyTorch and Huggingface

##Introduction:
The rise of Transformer-based architectures, particularly the likes of BERT and GPT, has revolutionized Natural Language Processing (NLP). One of the most significant advantages of these architectures is the ability to fine-tune pre-trained models on a specific task, such as text classification, with a smaller dataset. In this tutorial, we will delve deep into the process of fine-tuning a Transformer for this purpose, harnessing the power of PyTorch and the Huggingface library.

#Prerequisites:

A Python environment
Fundamental PyTorch knowledge
Basic understanding of NLP and Transformer models
#1. Why Fine-tuning?:
Large transformer models, when trained from scratch, require significant computational power and data. Fine-tuning leverages the general knowledge captured by pre-trained models and adjusts the weights to suit specific tasks, allowing us to achieve state-of-the-art results with much smaller datasets.

#2. Setting Up the Environment:
Start by installing the essential libraries:
"""

!pip install -q torch torchvision transformers[torch] datasets accelerate

"""#3. Choosing and Loading a Pre-trained Model:
Huggingface's transformers library offers a plethora of pre-trained models. For text classification, BERT (Bidirectional Encoder Representations from Transformers) is a solid choice due to its bidirectional nature, capturing context from both sides.
"""

from transformers import BertTokenizer, BertForSequenceClassification

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

"""num_labels=2 signifies that we are working with a binary classification problem.

#4. Data Acquisition and Preprocessing:
We'll use the IMDb dataset, a large movie review dataset for binary sentiment classification.
"""

from datasets import load_dataset

raw_datasets = load_dataset("imdb")

"""Tokenizing our dataset to convert text into input features:"""

def tokenize_function(examples):
    return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)

"""#5. Fine-tuning Mechanics:
Fine-tuning requires a slight adjustment to the model architecture, especially the top layers, to cater to the specific task (in our case, binary classification). The BertForSequenceClassification model automatically adjusts its head for classification tasks.

#6. Training Configuration and Fine-tuning:
Fine-tuning is just like regular training, but for fewer epochs, and with a learning rate smaller than usual since we don't want to deviate the pre-trained weights too much.
"""

from transformers import TrainingArguments, Trainer

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    output_dir = "./output",
    learning_rate=2e-5,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
)

trainer.train()

"""#7. Visualization and Interpretability:
Visualization is key for understanding the training dynamics. TensorBoard integration provided by the TrainingArguments makes it easy to monitor training:
"""

# Commented out IPython magic to ensure Python compatibility.
# %load_ext tensorboard
# %tensorboard --logdir ./logs

"""Beyond training metrics, understanding model predictions can be enhanced using libraries like Captum for PyTorch, providing insights into which parts of the input text influenced the model's decision.

#8. Evaluation and Deployment:
Post-training, evaluate the model to understand its performance:
"""

results = trainer.evaluate()
print(results)

"""For real-world applications, Huggingface's Pipelines offers a simple API to use the model for predictions, making deployment straightforward.

#9. Engaging Interaction - Live Predictions:
Interactive components can engage users and provide instant feedback. Using tools like Jupyter widgets, one can create interfaces for real-time predictions:
"""

import ipywidgets as widgets

def predict_review(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return "👍 Positive" if torch.argmax(probs) else "👎 Negative"

review_input = widgets.Textarea(
    value='',
    placeholder='Type a movie review...',
    description='Review:',
)

output = widgets.Label()

def on_value_change(change):
    output.value = predict_review(change['new'])

review_input.observe(on_value_change, 'value')
display(review_input, output)

"""#Conclusion:
Through this tutorial, we journeyed through the process of fine-tuning Transformers for text classification, highlighting the underlying concepts, practical PyTorch implementations, and ways to enhance user engagement. As NLP continues to evolve, so will the techniques and tools available, but the foundation laid here will remain relevant for years to come.
"""

