import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from random import shuffle
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import re
from tokenizer_functions import *
import pandas as pd
from utils import get_max_len_selfies, get_selfies_alphabet, add_onehot_noise
from typing import List
from sklearn.preprocessing import MinMaxScaler

# Define the hybrid chemical language model model
class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_output):
        # Apply attention
        attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        context_vector = attention_weights * lstm_output
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights

class RegressionHead(nn.Module):
    def __init__(self, d_embedding: int, output_size=1):
        super().__init__()
        self.layer1 = nn.Linear(d_embedding, d_embedding // 2)
        self.layer2 = nn.Linear(d_embedding // 2, d_embedding // 4)
        self.layer3 = nn.Linear(d_embedding // 4, d_embedding // 8)
        self.layer4 = nn.Linear(d_embedding // 8, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.layer1(x))
        x = self.relu(self.layer2(x))
        x = self.relu(self.layer3(x))
        return self.layer4(x)

class MolecularLSTMModel(nn.Module):
    """
    MolecularLSTMModel is a PyTorch module that implements a LSTM-based model for deep dreaming.

    Args:
        num_tokens_first_part (int): The number of tokens in the first part (one-hot encoded).
        num_tokens_second_part (int): The number of tokens in the second part (character-level embeddings).
        embedding_dim_second_part (int): The dimension of the embedding for the second part.
        hidden_dim (int): The dimension of the hidden state of the LSTM.
        output_dim (int): The dimension of the output.
        num_layers (int, optional): The number of LSTM layers. Defaults to 1.
        dropout_prob (float, optional): The dropout probability. Defaults to 0.1.
    """

    def __init__(self, 
                 num_tokens_first_part,
                 num_tokens_second_part,
                 embedding_dim_second_part,
                 hidden_dim, output_dim,
                 num_layers=1,
                 dropout_prob=0.1):
        super(MolecularLSTMModel, self).__init__()
        
        # LSTM for the first part (one-hot encoded)
        self.lstm1 = nn.LSTM(num_tokens_first_part, hidden_dim, num_layers, batch_first=True)
        self.attention1 = Attention(hidden_dim)
        self.dropout1 = nn.Dropout(p=dropout_prob)

        # LSTM for the second part (character-level embeddings)
        self.embedding2 = nn.Embedding(num_tokens_second_part, embedding_dim_second_part)
        self.lstm2 = nn.LSTM(embedding_dim_second_part, hidden_dim, num_layers, batch_first=True)
        self.attention2 = Attention(hidden_dim)
        self.dropout2 = nn.Dropout(p=dropout_prob)

        # Fully connected layer for regression
        # context vectors are the same length so can instantiate with a 2 * hidden_dim regardless of variable length. 
        self.regression_head = RegressionHead(2 * hidden_dim, output_dim)

    def forward(self, x1, x2, return_attention_weights=False):
        """
        Forward pass of the MolecularLSTMModel.

        Args:
            x1 (torch.Tensor): Input tensor for the first part.
            x2 (torch.Tensor): Input tensor for the second part.
            return_attention_weights (bool, optional): Whether to return attention weights. Defaults to False.

        Returns:
            torch.Tensor: The output tensor.
            torch.Tensor: The attention weights for the first part (if return_attention_weights is True).
            torch.Tensor: The attention weights for the second part (if return_attention_weights is True).
        """
        # Processing the first part
        lstm_output1, _ = self.lstm1(x1)
        context_vector1, attention_weights1 = self.attention1(lstm_output1)
        context_vector1 = self.dropout1(context_vector1)

        # Processing the second part
        x2 = self.embedding2(x2)
        lstm_output2, _ = self.lstm2(x2)                    # shape = [batch_size, seq_len, hidden_dim]
        context_vector2, attention_weights2 = self.attention2(lstm_output2)  # shape = [batch_size, hidden_dim]
        context_vector2 = self.dropout2(context_vector2)    # shape = [batch_size, hidden_dim]

        # Combining the context vectors from both parts
        combined = torch.cat((context_vector1, context_vector2), dim=1)

        # Passing the combined vector through the fully connected layer
        output = self.regression_head(combined)

        if return_attention_weights:
            return output, attention_weights1, attention_weights2
        else:
            return output

    def initialize_weights(self):
        """
        Initialize the weights of the LSTM and embedding layers.
        """
        # Initialize LSTM weights orthogonally
        for name, param in self.lstm1.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        for name, param in self.lstm2.named_parameters():
            if 'weight_ih' in name or 'weight_hh' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

        # Initialize embedding weights using Xavier
        nn.init.xavier_uniform_(self.embedding2.weight)

    
class HybridDataset(Dataset):
    def __init__(self, onehot_data, tokenized_data, targets):

        self.onehot_data = onehot_data
        self.tokenized_data = tokenized_data
        self.targets = targets
    """
    A class representing a hybrid dataset.

    Args:
        onehot_data (numpy.ndarray): The one-hot encoded data.
        tokenized_data (list): The tokenized data.
        targets (list): The target values.

    Attributes:
        onehot_data (numpy.ndarray): The one-hot encoded data.
        tokenized_data (list): The tokenized data.
        targets (list): The target values.
    """
    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.onehot_data[idx], self.tokenized_data[idx], self.targets[idx]


def featurize_df(df: pd.DataFrame, targets: List[str], edge_rep: str, node_rep: str, topo_rep: str): 
    """
    Featurizes the input DataFrame by extracting features from the specified columns
    and preparing the appropriate MOF string representations.

    Args:
        df (pd.DataFrame): The input DataFrame.
        targets (List[str]): The name of the target column.
        edge_rep (str): The name of the column containing edge representations.
        node_rep (str): The name of the column containing node representations.
        topo_rep (str): The name of the column containing topology representations.

    Returns:
        tuple: A tuple containing the featurized DataFrame and tokenized information.

    """
    to_featurize = df[['MOFname',edge_rep,node_rep,topo_rep]+targets]
    
    # edge data 
    max_len_selfie, edge_lens = get_max_len_selfies(to_featurize[edge_rep].tolist())
    alphabet = get_selfies_alphabet(to_featurize[edge_rep].tolist())
    symbol_to_idx = {symbol: i for i, symbol in enumerate(alphabet)}
    tokenized_selfies = tokenize_molecular_strings(
        to_featurize[edge_rep].tolist(),
        symbol_to_idx
        )
    
    to_featurize['tokenized_edge_selfies'] = tokenized_selfies
    to_featurize['edge_lens'] = edge_lens

    # node data    
    to_featurize['node_plus_topo'] = [a+f'[&&][{b}]' for a,b in zip(to_featurize[node_rep],to_featurize[topo_rep])]
    node_max_len_selfie, node_plus_topo_lens = get_max_len_selfies(to_featurize['node_plus_topo'].tolist())
    node_alphabet = get_selfies_alphabet(to_featurize['node_plus_topo'].tolist())
    node_symbol_to_idx = {symbol: i for i, symbol in enumerate(node_alphabet)}
    tokenized_node_plus_topo = tokenize_molecular_strings(
        to_featurize['node_plus_topo'].tolist(),
        node_symbol_to_idx,
        )
    to_featurize['tokenized_node_plus_topo'] = tokenized_node_plus_topo    
    to_featurize['node_plus_topo_lens'] = node_plus_topo_lens
    
    # get the combined string
    to_featurize['mof_string'] = [a+'[.]'+b+f'[&&][{c}]' for a,b,c in zip(to_featurize[edge_rep],to_featurize[node_rep],to_featurize[topo_rep])]
    mof_string_max_len_selfie, mof_string_lens = get_max_len_selfies(to_featurize['mof_string'].tolist())
    mof_string_alphabet = get_selfies_alphabet(to_featurize['mof_string'].tolist())
    mof_string_symbol_to_idx = {symbol: i for i, symbol in enumerate(mof_string_alphabet)}
    tokenized_mof_string = tokenize_molecular_strings(
        to_featurize['mof_string'].tolist(),
        mof_string_symbol_to_idx,
        )
    to_featurize['tokenized_mof_string'] = tokenized_mof_string    
    to_featurize['mof_string_lens'] = mof_string_lens

    tokenized_info = {'max_len_selfie': max_len_selfie,
                    'alphabet': alphabet,
                    'symbol_to_idx': symbol_to_idx,
                    'node_plus_topo_max_len_selfie': node_max_len_selfie,
                    'node_plus_topo_alphabet': node_alphabet,
                    'node_plus_topo_symbol_to_idx': node_symbol_to_idx,
                    'mof_string_max_len_selfie': mof_string_max_len_selfie,
                    'mof_string_alphabet': mof_string_alphabet,
                    'mof_string_symbol_to_idx': mof_string_symbol_to_idx,
                    }
    
    return to_featurize, tokenized_info


def prepare_hybrid_dataset(df,
                           tokenized_info,
                           target_names: List[str],
                           pad_node=True,
                           batch_size: int=1,
                           shuffle=True,
                           train=True,
                           scale_targets=True,
                           scaler=None,
                           noise_level: float=None,
                           seed: int=None):
    """
    Prepares a hybrid dataset for training or testing a model.

    Args:
        df (pandas.DataFrame): The input dataframe containing the dataset (prepared using featurize_df function)
        tokenized_info (dict): Information about the tokenized sequences.
        target_names (List[str]): The target variables to be predicted.
        pad_node (bool, optional): Whether to pad the node sequences. Defaults to True.
        flag (str, optional): A flag indicating the type of dataset. Defaults to 'normal'.
        batch_size (int, optional): The batch size for the DataLoader. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to True.
        train (bool, optional): Whether the dataset is for training. Defaults to True.
        scale_targets (bool, optional): Whether to scale the target variable. Defaults to True.
        scaler (object, optional): The scaler object for scaling the target variable. Defaults to None.
        noise_level (float, optional): The level of noise to add to the input. Defaults to None.
        seed (int, optional): Random noise seed (for reproducbility)
    Returns:
        torch.utils.data.DataLoader: The DataLoader object containing the prepared dataset.
        object: The scaler object used for scaling the target variable.
    """

    padded_tokenized_encoding = [pad_tokenized_sequence(
        df['tokenized_edge_selfies'].iloc[i],
        max_sequence_length=tokenized_info['max_len_selfie'],
        padding_value=tokenized_info['symbol_to_idx']['[nop]']
        ) 
        for i in range(len(df['tokenized_edge_selfies']))]
    padded_onehot_encoding = [one_hot_encode(padded_tokens, len(tokenized_info['alphabet'])) for padded_tokens in padded_tokenized_encoding]
    # padded_onehot_encoding = [one_hot_encode(df['tokenized_edge_selfies'].iloc[i], len(tokenized_info['alphabet'])) for i in range(len(df['tokenized_edge_selfies']))]
    if noise_level is not None:
        onehot_input = add_onehot_noise(padded_onehot_encoding, noise_level, seed=seed)
    else:
        onehot_input = torch.tensor(np.stack(padded_onehot_encoding),dtype=torch.float32)

    if pad_node:
        embedding_encoding = [pad_tokenized_sequence(
            df['tokenized_node_plus_topo'].iloc[i],
            max_sequence_length=tokenized_info['node_plus_topo_max_len_selfie'],
            padding_value=tokenized_info['node_plus_topo_symbol_to_idx']['[nop]']
            )
            for i in range(len(df['tokenized_node_plus_topo']))]
        tokenized_input = torch.tensor(np.stack(embedding_encoding),dtype=torch.long) 
    else:
        tokenized_input = [df['tokenized_node_plus_topo'].iloc[i] for i in range(len(df['tokenized_node_plus_topo']))]

    if train:
        if scale_targets:
            scaler = MinMaxScaler()
            if len(target_names) == 1:
                scaled_targets = scaler.fit_transform(np.stack(df[target_names[0]]).reshape(-1,1))
            else:
                stacked_targets = np.stack([df[name] for name in target_names], axis=1)
                scaled_targets = scaler.fit_transform(stacked_targets)
        else:
            scaled_targets = np.stack([df[name] for name in target_names], axis=1)
    else:
        if scaler is None and scale_targets:
            raise ValueError('Scaler must be provided for test dataset')
        if len(target_names) == 1:
            scaled_targets = scaler.transform(np.stack(df[target_names[0]]).reshape(-1,1))
        else:
            stacked_targets = np.stack([df[name] for name in target_names], axis=1)
            scaled_targets = scaler.transform(stacked_targets)
    print(scaled_targets.shape)
    targets = torch.tensor(scaled_targets,dtype=torch.float32).squeeze(1)
    dataset = HybridDataset(onehot_input, tokenized_input, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), scaler


def prepare_dreaming_mof(mof_strings: list,
                           tokenized_info: dict,
                           pad_node = True,
                           noise_level: float = None,
                           seed: int=None):
    """
    Prepares the input mof string for the dreaming process.

    Args:
        mof_strings (list): A list of MOF strings.
        tokenized_info (dict): Information about the tokenization process.
        pad_node (bool, optional): Whether to pad the node_plus_topo sequences. Defaults to True.
        noise_level (float, optional): The level of noise to add to the input. Defaults to None.
        seed (int, optional): Random noise seed (for reproducibility)
    Returns:
        tuple: A tuple containing the one-hot encoded input and the tokenized input.
    """

    # Tokenize the edge
    selfies = [mof.split('[.]')[0] for mof in mof_strings]
    tokenized_selfies = tokenize_molecular_strings(
        selfies,
        tokenized_info['symbol_to_idx'], 
        )
    
    # tokenize the node_plus_topo
    node_plus_topos = [mof.split('[.]')[1] for mof in mof_strings]
    tokenized_node_plus_topo = tokenize_molecular_strings(
        node_plus_topos,
        tokenized_info['node_plus_topo_symbol_to_idx'], 
        )

    # pad sequences if needed  and then one-hot encode
    padded_tokenized_encoding = [pad_tokenized_sequence(
        selfie,
        max_sequence_length=tokenized_info['max_len_selfie'],
        padding_value=tokenized_info['symbol_to_idx']['[nop]']
        ) 
        for selfie in tokenized_selfies]
    padded_onehot_encoding = [one_hot_encode(padded_tokens, len(tokenized_info['alphabet'])) for padded_tokens in padded_tokenized_encoding]
    if noise_level is not None:
        onehot_input = add_onehot_noise(padded_onehot_encoding, noise_level, seed=seed)
    else:
        onehot_input = torch.tensor(np.stack(padded_onehot_encoding),dtype=torch.float32)

    # print(onehot_input.shape)
    if pad_node:
        embedding_encoding = [pad_tokenized_sequence(
            node_plus_topo,
            max_sequence_length=tokenized_info['node_plus_topo_max_len_selfie'],
            padding_value=tokenized_info['node_plus_topo_symbol_to_idx']['[nop]']
            )
            for node_plus_topo in tokenized_node_plus_topo]
        tokenized_input = torch.tensor(np.stack(embedding_encoding),dtype=torch.long) 
    else:
        tokenized_input = tokenized_node_plus_topo
        
    return torch.autograd.Variable(onehot_input, requires_grad=True), torch.tensor(tokenized_input,dtype=torch.long)


def prepare_dreaming_edge(selfies: list,
                           tokenized_info: dict,
                           noise_level: float = None,
                           seed: int=None):
    """
    Prepares the input edge string for the dreaming process.

    Args:
        selfies (list): A list of edge strings.
        tokenized_info (dict): Information about the tokenization process.
        noise_level (float, optional): The level of noise to add to the input. Defaults to None.
        seed (int, optional): Random noise seed (for reprodcibility)
    Returns:
        tuple: A tuple containing the one-hot encoded input and the tokenized input.
    """

    # Tokenize the edge
    tokenized_selfies = tokenize_molecular_strings(
        selfies,
        tokenized_info['symbol_to_idx'], 
        )

    # pad sequences if needed  and then one-hot encode
    padded_tokenized_encoding = [pad_tokenized_sequence(
        selfie,
        max_sequence_length=tokenized_info['max_len_selfie'],
        padding_value=tokenized_info['symbol_to_idx']['[nop]']
        ) 
        for selfie in tokenized_selfies]
    padded_onehot_encoding = [one_hot_encode(padded_tokens, len(tokenized_info['alphabet'])) for padded_tokens in padded_tokenized_encoding]
    if noise_level is not None:
        onehot_input = add_onehot_noise(padded_onehot_encoding, noise_level, seed=seed)
    else:
        onehot_input = torch.tensor(np.stack(padded_onehot_encoding),dtype=torch.float32)
        
    return torch.autograd.Variable(onehot_input, requires_grad=True)


def validate(model, test_loader, criterion, model_type='dreaming'):
    """
    Validates the performance of a model on a given test dataset.

    Args:
        model (torch.nn.Module): The model to be validated.
        test_loader (torch.utils.data.DataLoader): The data loader for the test dataset.
        criterion (torch.nn.Module): The loss function used for evaluation.

    Returns:
        tuple: A tuple containing the mean loss and mean absolute error (MAE) of the model on the test dataset.
    """
        
    model.eval()
    total_loss = 0
    total_mae = 0

    device = next(model.parameters()).device  # Get the device of the model

    if model_type == 'dreaming':
        with torch.no_grad():
            for i, (onehot_input, embedding_input, target_values) in enumerate(test_loader):
                if type(embedding_input) == list:
                    embedding_input = torch.stack(embedding_input, dim=0).squeeze().unsqueeze(0)    
                onehot_input, embedding_input, target_values = onehot_input.to(device), embedding_input.to(device), target_values.to(device)
                output = model(onehot_input, embedding_input)
                total_loss += criterion(output.squeeze(), target_values).item()
                total_mae += torch.abs(output - target_values).sum().item()
    elif model_type == 'predictor':
        with torch.no_grad():
            for i, (input, target_values) in enumerate(test_loader):
                if type(input) == list:
                    input = torch.stack(input, dim=0).squeeze().unsqueeze(0)    
                input, target_values = input.to(device), target_values.to(device)
                output = model(input)
                total_loss += criterion(output.squeeze(), target_values).item()
                total_mae += torch.abs(output - target_values).sum().item()
    else:
        raise ValueError('model_type must be dreaming or predictor')

    mean_loss = total_loss / len(test_loader)
    mean_mae = total_mae / len(test_loader.dataset)
    model.train()  # put model back in training mode
    return mean_loss, mean_mae


def split_dataframe(df: pd.DataFrame, splitting_method: dict, seed=None):
    """
    Split a dataframe into train, validate, and test sets based on the specified splitting method.

    Args:
    - df (pandas.DataFrame): The dataframe to be split.
    - splitting_method (dict): A dictionary specifying the splitting method. It should contain the following keys:
        - 'method' (str): The splitting method to be used. Valid options are 'ratio' and 'sample'.
        - 'train' (float or int): The proportion or absolute size of the training set.
        - 'validate' (float or int): The proportion or absolute size of the validation set.
        - 'test' (float or int): The proportion or absolute size of the test set.
    - seed (int, optional): The random seed for shuffling the dataframe. Default is None.

    Returns:
    - train_set (pandas.DataFrame): The training set.
    - validate_set (pandas.DataFrame): The validation set.
    - test_set (pandas.DataFrame): The test set.

    Raises:
    - ValueError: If an invalid splitting method is specified.
    """
    from sklearn.utils import shuffle
    df = shuffle(df, random_state=seed)

    if splitting_method['method'] == 'ratio':
        train_size = int(len(df) * splitting_method['train'])
        validate_size = int(len(df) * splitting_method['validate'])
        test_size = int(len(df) * splitting_method['test'])
    elif splitting_method['method'] == 'sample':
        train_size = splitting_method['train']
        validate_size = splitting_method['validate']
        test_size = splitting_method['test']
    else:
        raise ValueError('Invalid splitting method.')
    train_set = df.iloc[:train_size]
    validate_set = df.iloc[train_size:train_size+validate_size]
    test_set = df.iloc[train_size+validate_size:min(test_size+train_size+validate_size, len(df))]

    train_set.reset_index(inplace=True, drop=True)
    validate_set.reset_index(inplace=True, drop=True)
    test_set.reset_index(inplace=True, drop=True)
    return train_set, validate_set, test_set