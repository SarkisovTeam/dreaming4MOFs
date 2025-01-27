import os, sys
import pandas as pd
import selfies as sf
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Fingerprints import FingerprintMols
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
## TODO: REMOVE DEPENDENCY ON TORCH HERE 
import torch
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from scipy.stats import expon, truncnorm
import copy
from collections import defaultdict
import math
constraints = {'H': 1, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'B': 3, 'B+1': 2, 'B-1': 4, 'O': 2, 'O+1': 3, 'O-1': 1, 'N': 3, 'N+1': 4, 'N-1': 2, 'C': 4, 'C+1': 5, 'C-1': 3, 'P': 5,
               'P+1': 6, 'P-1': 4, 'S': 6, 'S+1': 7, 'S-1': 5, '?': 8, 'Fr': 1}
sf.set_semantic_constraints(constraints)


#******************************************************************************
# UTILITY FUNCTIONS 
#******************************************************************************
def get_key(my_dict,val):
    '''
    get key of a dictionary by passing in the value
    '''
    for key, value in my_dict.items():
        if val == value:
            return key
 
    return "key doesn't exist"


def get_molecule_composition(smiles):
    '''
    Get molecular composition of a molecule described by a smiles string
    '''
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    # Initialize a defaultdict to store element counts
    composition = defaultdict(int)
    
    # Iterate through the atoms in the molecule
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        composition[symbol] += 1
    
    return dict(composition)


def calculate_atomic_mass(smiles_string):
    '''
    Get atomic mass from a smiles string
    '''
    molecule = Chem.MolFromSmiles(smiles_string)
    return Chem.rdMolDescriptors.CalcExactMolWt(molecule)


def tanimoto_similarity(fp1, fp2):
    """
    Compute Tanimoto similarity between two fingerprints.
    """
    return DataStructs.FingerprintSimilarity(fp1, fp2)



def compute_fingerprints(smiles_list, radius=2, nBits=1024):
    """
    Compute Morgan fingerprints for a list of SMILES strings.
    """
    fingerprints = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits)
            fingerprints.append(fp)
    return fingerprints

    
def snn_metric(generated_smiles, reference_smiles, return_similarities=False):
    """
    Compute the Similarity to Nearest Neighbor (SNN) metric.
    """
    generated_fps = compute_fingerprints(generated_smiles)
    reference_fps = compute_fingerprints(reference_smiles)

    max_similarities = []
    for g_fp in generated_fps:
        max_similarity = max(tanimoto_similarity(g_fp, r_fp) for r_fp in reference_fps)
        max_similarities.append(max_similarity)

    if return_similarities:
        return np.mean(max_similarities), max_similarities
    else:
        return np.mean(max_similarities)

def category_to_vec(cat_var: list):
    ''''
    Take a list of categorical variables and create a dictionary
    '''
    unique_entries = list(set(cat_var))
    unique_entries.sort()
    return {cat: i for i, cat in enumerate(unique_entries)}

# def add_onehot_noise(x: list, k: float):
#     """
#     Adds one-hot noise to a one-hot encoded tensor.

#     Args:
#         x (list): The input list of tensors.
#         k (float): The settings for the one-hot noise.

#     Returns:
#         torch.Tensor: The tensor with one-hot noise added.
#     """
#     input_tensor = torch.stack(x, dim=0)
#     noise = torch.rand(input_tensor.shape) * k
#     mask = (input_tensor == 0)
#     noisy_tensor = input_tensor + noise * mask
#     normalized_tensor = noisy_tensor / torch.sum(noisy_tensor, dim=2, keepdim=True)

#     return normalized_tensor

def add_onehot_noise(x: list, k: float, seed: int = None):
    """
    Adds one-hot noise to a one-hot encoded tensor.

    Args:
        x (list): The input list of tensors.
        k (float): The settings for the one-hot noise.
        seed (int): Random noise seed (for reproducbility)

    Returns:
        torch.Tensor: The tensor with one-hot noise added.
    """
    gen = None
    if seed is not None:
        gen = torch.Generator()
        gen.manual_seed(seed)

    input_tensor = torch.stack(x, dim=0)
    # Pass the local generator to torch.rand if provided
    if gen is not None:
        noise = torch.rand(input_tensor.shape, generator=gen) * k
    else:
        noise = torch.rand(input_tensor.shape) * k

    mask = (input_tensor == 0)
    noisy_tensor = input_tensor + noise * mask
    normalized_tensor = noisy_tensor / torch.sum(noisy_tensor, dim=2, keepdim=True)

    return normalized_tensor

def calculate_max_length(smi):
    '''
    Calculates the maximium size of a linker in angstroms
    '''

    mol = Chem.MolFromSmiles(smi)
    mol = Chem.AddHs(mol)
    Chem.AllChem.EmbedMolecule(mol, randomSeed=42)
    Chem.AllChem.MMFFOptimizeMolecule(mol)
    max_length = 0
    conf = mol.GetConformer()
    num_atoms = mol.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i + 1, num_atoms):
            distance = Chem.rdMolTransforms.GetBondLength(conf, i, j)
            if distance > max_length:
                max_length = distance
    return max_length


def canonicalize_selfies(selfies):
    """
    Canonicalizes a SELFIES string.
    """
    smiles = sf.decoder(selfies)
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


def calculate_diversity(smiles_list ,n_samples, seed=42):
    picker = MaxMinPicker()
    fingerprints = compute_fingerprints(smiles_list)
    return list(picker.LazyBitVectorPick(fingerprints, len(fingerprints), n_samples, seed=seed))



#******************************************************************************
# ENCODING FUNCTIONS 
#******************************************************************************
def smiles_linkers_to_selfies(list_of_linkers: list) -> list:
    """
    Converts a list of SMILES linkers to their corresponding SELFIES representation.

    Args:
        list_of_linkers (list): A list of SMILES linkers.

    Returns:
        list: A list of SELFIES representations of the input SMILES linkers.
    """

    list_of_selfies = []
    for linker in list_of_linkers:
        try:
            linker_sf = sf.encoder(linker)
        except:
            # return encoding error if failure
            linker_sf = np.nan
        
        list_of_selfies.append(linker_sf)
    
    return list_of_selfies


def get_selfies_alphabet(selfies_list: list) -> list:
    """
    Get the alphabet of SELFIES strings from a given list of Selfies.

    Args:
        selfies_list (list): A list of SELFIES strings.

    Returns:
        list: The alphabet of SELFIES strings, including a padding entry '[nop]'.

    """
    # get rid of any 'NaN' entries in the selfies_list
    cleaned_selfies_list = [item for item in selfies_list if not(pd.isnull(item)) == True]
    
    alphabet = sf.get_alphabet_from_selfies(cleaned_selfies_list)
    
    # add padding entries - [nop] i.e., no operation, is a special 
    # symbol to pad selfies strings and is always skipped in the 
    # selfies.decoder function

    return ['[nop]'] + list(sorted(alphabet))


def get_max_len_selfies(list_of_selfies: list) -> int:
    """
    Calculates the maximum length of SELFIES in a given list.

    Args:
        list_of_selfies (list): A list of SELFIES.

    Returns:
        int: The maximum length of SELFIES.
        list: A list of the lengths of each SELFIES string in the input list.

    """
    len_selfies = [sf.len_selfies(selfie) for selfie in list_of_selfies]
    return max(len_selfies), len_selfies


def selfies_to_one_hot(selfie_str: str, symbol_to_idx, pad_to_len: int):
    """
    Converts a SELFIES string to a one-hot encoding.

    Args:
        selfie_str (str): The SELFIES string to be converted.
        symbol_to_idx (dict): A dictionary mapping SELFIES symbols to indices.
        pad_to_len (int): The length to which the one-hot encoding should be padded.

    Returns:
        tuple: A tuple containing the label (integer between 0 and len(alphabet)),
               the one-hot encoding of the SELFIES string, and the concatenated one-hot encoding.
    """
    label, one_hot = sf.selfies_to_encoding(
        selfies=selfie_str,
        vocab_stoi=symbol_to_idx,
        pad_to_len=pad_to_len,
        enc_type="both"
    )

    return label, np.array(one_hot), np.concatenate(np.array(one_hot))


def multiple_selfies_to_hot(selfies_list, pad_to_len, symbol_to_idx):
    """
    Convert a list of SELFIES strings to their corresponding one-hot encodings.

    Args:
        selfies_list (list): A list of SELFIES strings.
        pad_to_len (int): The length to which each SELFIES string should be padded.
        symbol_to_idx (dict): A dictionary mapping each SELFIES symbol to its index.

    Returns:
        tuple: A tuple containing three elements:
            - np.array: An array of one-hot encodings for each SELFIES string.
            - np.array: An array of concatenated one-hot encodings for each SELFIES string.
            - int: The length of the concatenated one-hot encoding.

    """
    # get list of one-hot encodings
    hot_list = []
    concat_hot_list = []
    for selfie_str in selfies_list:
        _, onehot_encoded, onehot_concat = selfies_to_one_hot(selfie_str=selfie_str,
                                                              symbol_to_idx=symbol_to_idx,
                                                              pad_to_len=pad_to_len)
        hot_list.append(onehot_encoded)
        concat_hot_list.append(onehot_concat)
    return np.array(hot_list), np.array(concat_hot_list), len(onehot_concat)


#******************************************************************************
# DECODING FUNCTIONS 
#******************************************************************************    

def onehot_to_alphabet_idx(onehot: torch.Tensor, alphabet: dict, max_len_selfie: int, concatenated=True):
    """
    Converts a one-hot encoded tensor to a list of alphabet indices.

    Args:
        onehot (torch.Tensor): The one-hot encoded tensor.
        alphabet (dict): A dictionary mapping alphabet characters to indices.
        max_len_selfie (int): The maximum length of the selfie.
        concatenated (bool, optional): Whether the onehot tensor is concatenated or not. 
            Defaults to True.

    Returns:
        list: A list of alphabet indices.

    """
    lst_of_idx = [] 
    if concatenated:
        onehot = torch.reshape(onehot, (max_len_selfie, len(alphabet))).detach().numpy()
    else:
        onehot = onehot.detach().numpy()
    for entry in onehot:
        lst_of_idx.append(entry.argmax())
    
    return lst_of_idx


def alphabet_index_to_selfies(list_of_indexs: list, alphabet: dict, nop_index=0):
    """
    Converts a list of alphabet indices to a selfies string.

    Args:
        list_of_indexs (list): A list of alphabet indices.
        alphabet (dict): The alphabet used for conversion.
        nop_index (int, optional): The index representing the [nop] padding entry. Defaults to 0.

    Returns:
        str: The selfies string converted from the list of indices.
    """

    selfies = ''
    for index in list_of_indexs:
        if index != nop_index:
            # skip the [nop] padding entry in the selfies alphabet
            selfies += alphabet[index]

    return selfies


def onehot_to_selfies(onehot_input: torch.Tensor, alphabet: dict, max_len_selfie: int, nop_index: int = 0):
    """
    Converts a one-hot encoded input to SELFIES representation.

    Args:
        onehot_input (torch.Tensor): The one-hot encoded input.
        alphabet (dict): The alphabet used for encoding.
        max_len_selfie (int): The maximum length of the SELFIES representation.
        nop_index (int, optional): The index of the 'no operation' symbol in the alphabet. Defaults to 0.

    Returns:
        tuple: A tuple containing the SELFIES representation and the tokenized version of the input.

    """
    tokens = onehot_to_alphabet_idx(
        onehot_input,
        alphabet,
        max_len_selfie,
        concatenated=True
    )
    return alphabet_index_to_selfies(tokens, alphabet, nop_index=nop_index), tokens


def onehot_group_selfies_to_smiles(onehot_input, alphabet: int, max_len_selfie: int, grammar, nop_index: int = 0):
    """
    Converts a one-hot encoded input to SMILES representation using SELFIES decoding.

    Args:
        onehot_input (torch.Tensor): The one-hot encoded input.
        alphabet (dict): The size of the alphabet used for encoding.
        max_len_selfie (int): The maximum length of the SELFIES representation.
        grammar (obj): The group SELFIES grammar object.
        nop_index (int): The index of the "no operation" symbol in the alphabet.

    Returns:
        mol: The molecule object obtained from SELFIES decoding.
        smiles: The SMILES representation of the molecule.

    """
    selfies, _ = onehot_to_selfies(onehot_input, alphabet, max_len_selfie, nop_index)
    mol = grammar.decoder(selfies)
    return mol, Chem.MolToSmiles(mol)
