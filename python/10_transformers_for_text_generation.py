# -*- coding: utf-8 -*-
"""
# Transformers for Text Generation Tutorial with Python, PyTorch, and Hugging Face

## Introduction
Transformers have revolutionized the field of natural language processing. They are powerful models that have achieved state-of-the-art results in various tasks, including text generation. This tutorial covers the implementation of a text generation model using Python, PyTorch, and Hugging Face's Transformers library.

# Setup
Before diving into the code, you need to set up your environment. Run the following commands in a Google Colab notebook to install the necessary packages:
"""

!pip install torch transformers

"""After installing, check the versions:"""

import torch
import transformers

print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")

"""## Understanding Transformers
Transformers are a type of neural network architecture that primarily use attention mechanisms to understand the context of a given input. Unlike previous models that processed data sequentially, transformers process data in parallel, making them faster and more efficient.

## Key Components:
* Attention Mechanisms: Help the model focus on relevant parts of the input.
* Encoder-Decoder Architecture: Common in many transformer models, with encoders processing the input and decoders generating the output.

## Implementing Text Generation
We'll use a pre-trained model from the Hugging Face library for text generation. One popular model is GPT-2, known for its effectiveness in generating coherent and contextually relevant text.

## Importing Libraries
"""

from transformers import GPT2LMHeadModel, GPT2Tokenizer

"""## Loading the Pre-Trained Model"""

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

"""## Function for Generating Text"""

def generate_text(prompt, length=100, temperature=1.0, k=50, p=0.95):
    """
    Generates text using the GPT-2 model.

    :param prompt: The initial text to start generation.
    :param length: The length of the generated text.
    :param temperature: Controls the randomness of predictions.
    :param k: The K most likely next words are filtered.
    :param p: Nucleus sampling's probability threshold.
    :return: Generated text.
    """
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    output = model.generate(input_ids, max_length=length, temperature=temperature, top_k=k, top_p=p)
    return tokenizer.decode(output[0], skip_special_tokens=True)

"""## Generating Text"""

prompt = "Once upon a time"
generated_text = generate_text(prompt)
print(generated_text)

"""## Visualization
Visualizing the attention mechanisms or the internal workings of the model can be quite complex. However, tools like BertViz can be used for such visualizations. Unfortunately, detailed implementation of such visualizations is beyond the scope of this tutorial.

## Reproducibility
To ensure reproducibility:

* Specify the model version when loading it.
* Use a fixed seed for random number generators.
"""

torch.manual_seed(0)

"""## Conclusion
This tutorial provided an overview and implementation of a transformer model for text generation. Transformers, with their parallel processing and attention mechanisms, offer great power and efficiency in natural language tasks.

## Version Information (Run at the End)
Make sure to run this cell at the end of your experimentation to log the version information:


"""

print(f"PyTorch Version: {torch.__version__}")
print(f"Transformers Version: {transformers.__version__}")

"""This tutorial aimed at providing a comprehensive yet accessible introduction to using transformers for text generation. The code is designed to be run end-to-end in a Google Colab notebook, ensuring ease of use and accessibility for learners and enthusiasts."""

