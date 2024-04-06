# Install necessary libraries
!pip install transformers torch
# Importing necessary libraries
from transformers import BertTokenizer, BertModel

# Loading the pre-trained BERT model and tokenizer
model_name = "bert-base-uncased"
model = BertModel.from_pretrained(model_name)
tokenizer = BertTokenizer.from_pretrained(model_name)

print(f"Loaded {model_name} model and tokenizer.")
# Sample text
text = "Hello, world! This is an introduction to transformers."

# Tokenizing the text
tokens = tokenizer.tokenize(text)
token_ids = tokenizer.convert_tokens_to_ids(tokens)

print(f"Tokens: {tokens}")
print(f"Token IDs: {token_ids}")
# Encoding the text and getting model outputs
inputs = tokenizer.encode_plus(text, return_tensors="pt", add_special_tokens=True)
outputs = model(**inputs)

# Extracting the last hidden state (features)
last_hidden_state = outputs.last_hidden_state

print(f"Shape of the last hidden state: {last_hidden_state.shape}")
