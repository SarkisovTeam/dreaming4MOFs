from typing import List
import re
import torch

def get_tokens(string):
    '''
    Returns a list of tokens from a SELFIES string
    '''
    return re.findall('\[.*?\]|.', string)

def prepare_input_for_language_model(tokens, token_to_index):
    # Convert tokens to indices based on a token-to-index mapping
    token_indices = [token_to_index[token] for token in tokens]
    return token_indices

def pad_tokenized_sequence(token_indices, max_sequence_length, padding_value=0):
    if len(token_indices) < max_sequence_length:
        token_indices += [padding_value] * (max_sequence_length - len(token_indices))
    else:
        token_indices = token_indices[:max_sequence_length]
    
    return token_indices

# Function to one-hot-encode a sequence (assuming a fixed number of tokens)
def one_hot_encode(sequence, num_tokens):
    return torch.eye(num_tokens)[sequence]

def tokenize_molecular_strings(molecular_strings: List[str], symbol_to_idx: dict):
    tokenized_molecular_strings = []
    for molecular_string in molecular_strings:
        tokens = get_tokens(molecular_string)
        tokenized_molecular_string = prepare_input_for_language_model(tokens, symbol_to_idx)
        tokenized_molecular_strings.append(tokenized_molecular_string)
    return tokenized_molecular_strings
