import torch.nn as nn
from torch import optim
import numpy as np
from utils import onehot_to_selfies, get_molecule_composition, onehot_group_selfies_to_smiles
from nn_functions import prepare_dreaming_mof, prepare_dreaming_edge
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdMolTransforms import GetBondLength
import selfies as sf
import pandas as pd
import sys, os
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import sascorer
from ase.io import read, write
from io import StringIO
from ase.visualize import view
from kpi_small_mols import SAscore_from_smiles
from typing import List
from scscorer import SCScorer


def check_token_overlap(tokens, alphabet):
    return all(token in alphabet for token in tokens)


def SA_score_penalty(perturbed_structure, penalty_scaler=0.1):
    """
    Calculates the SA score penalty for a perturbed structure.

    Parameters:
    - perturbed_structure (str): The perturbed structure in SMILES format.
    - penalty_scaler (float): A scaling factor for the penalty score (default: 0.1).

    Returns:
    - float: The SA score penalty for the perturbed structure.
    """
    mol = Chem.MolFromSmiles(perturbed_structure)
    return sascorer.calculateScore(mol) * penalty_scaler


def connection_penalty(perturbed_structure, penalty_per_connection=0.1):
    """
    Calculates the penalty for the number of connections in a perturbed structure.

    Parameters:
    - perturbed_structure: The perturbed structure for which the connection penalty is calculated.
    - penalty_per_connection: The penalty value assigned per connection.

    Returns:
    - The connection penalty for the perturbed structure.

    The function extracts the number of Francium atoms (pseudoatoms for connection points) in the molecule
    and applies a penalty based on the difference between the count and the expected value of 2. If the count
    is exactly 2, the penalty is 0.0.
    """
    # Extract the number of Francium atoms (pseudoatoms for connection points) in the molecule
    composition = get_molecule_composition(perturbed_structure)
    try:
        Fr_count = composition['Fr']
    except: 
        Fr_count = 0
    # Apply a heavy penalty if the count is not 2
    if Fr_count != 2:
        return penalty_per_connection * abs(Fr_count - 2) 
    else:
        return 0.0
    

def connection_point_graph_distance(smiles, connection='Fr'):
    """
    Calculates the normalized distance between two placeholder atoms in a molecule.

    Parameters:
        smiles (str): The SMILES representation of the molecule.
        connection (str, optional): The symbol of the placeholder atom. Defaults to 'Fr'.

    Returns:
        float: The normalized distance between the two placeholder atoms.
    """
    # Find indices of the placeholder atoms
    mol = Chem.MolFromSmiles(smiles)
    placeholder_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetSymbol() == connection] 

    # Calculate the shortest path between the two placeholder atoms
    shortest_path_length = len(Chem.rdmolops.GetShortestPath(mol, placeholder_indices[0], placeholder_indices[1])) - 1

    # Calculate the maximum path length within the molecule (diameter of the graph)
    all_pairs = Chem.rdmolops.GetDistanceMatrix(mol)
    max_path_length = int(all_pairs.max())

    # Normalize the shortest path length by the maximum path length
    normalized_distance = shortest_path_length / max_path_length
    return normalized_distance


def connection_point_atomic_distance(smiles, connection='Fr', visualise=False):
    """
    Calculates the normalized distance and the actual distance between two placeholder atoms in a molecule.

    Parameters:
    - smiles (str): The SMILES representation of the molecule.
    - connection (str): The atomic symbol of the placeholder atom. Default is 'Fr'.
    - visualise (bool): Whether to visualize the molecule using ASE. Default is False.

    Returns:
    - normalized_distance (float): The normalized distance between the placeholder atoms.
    - distance_between_placeholders (float): The actual distance between the placeholder atoms.

    Raises:
    - AssertionError: If there are not exactly two placeholder atoms in the molecule.
    """
    molecule = Chem.MolFromSmiles(smiles)
    # Add hydrogens (if necessary) and generate a 3D conformation
    molecule = Chem.AddHs(molecule)
    AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(molecule)

    if visualise:
        ase_molecule = Chem.MolToMolBlock(molecule)
        ase_atoms = read(StringIO(ase_molecule), format='mol')
        view(ase_atoms)

    # Identify the placeholder atoms by their atomic number
    placeholder_atomic_number = Chem.Atom(connection).GetAtomicNum() 
    placeholder_atom_indices = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == placeholder_atomic_number]

    # Ensure that there are exactly two placeholder atoms
    assert len(placeholder_atom_indices) == 2, "There should be exactly two placeholder atoms."

    # Calculate the distance between the two placeholder atoms
    distance_between_placeholders = GetBondLength(molecule.GetConformer(), placeholder_atom_indices[0], placeholder_atom_indices[1])

    # Find the maximum distance between any two atoms in the molecule
    max_distance = 0
    num_atoms = molecule.GetNumAtoms()
    for i in range(num_atoms):
        for j in range(i+1, num_atoms):  # Only look at unique pairs
            distance = GetBondLength(molecule.GetConformer(), i, j)
            max_distance = max(max_distance, distance)

    # Normalize the distance between the placeholder atoms
    normalized_distance = distance_between_placeholders / max_distance
    return normalized_distance, distance_between_placeholders


def sc_score_penalty(perturbed_structure, model):
    """
    Calculates the SC score penalty for a perturbed structure based on a given model.

    Parameters:
    - perturbed_structure (str): The perturbed structure to calculate the score penalty for.
    - model: The model used to calculate the score.

    Returns:
    - score_penalty (float): The score penalty for the perturbed structure.
    """
    (smi_conv, sco) = model.get_score_from_smi(perturbed_structure)
    if sco >= 4:
        return 100 * sco
    elif (sco > 3) & (sco < 4):
        return 10 * sco
    else:
        return sco


# def dream(
#         model,
#         predictor,
#         seed_mof_string: str, 
#         target_values: List[float], 
#         tokenized_info: dict, 
#         group_grammar,
#         dream_settings: dict,
#         ):
#     """
#     Generates new MOFs using a dreaming model.

#     Args:
#         model (nn.Module): The dreaming model used for generating new MOFs.
#         predictor (nn.Module): The predictor model used for evaluating the generated MOFs.
#         seed_mof_string (str): The seed MOF structure used as a starting point for dreaming.
#         target_values (float): The target values that the dreaming model aims to achieve.
#         tokenized_info (dict): Tokenized information used for preparing the input.
#         group_grammar: The group grammar used for encoding and decoding the MOF structures.
#         dream_settings (dict): Settings for the dreaming process.

#     Returns:
#         pd.DataFrame: DataFrame containing the valid linker optimization pathway.
#         pd.DataFrame: DataFrame containing the molecule transmutation pathway.
#         dict: Dictionary containing the targets, losses, epochs, and early stop flag during training.
#     """
        
#     # model settings
#     model.eval() 
#     predictor.eval()
#     criterion = nn.MSELoss()
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # tokenized info
#     alphabet = tokenized_info['alphabet']
#     max_len_selfie = tokenized_info['max_len_selfie']
#     symbol_to_idx = tokenized_info['symbol_to_idx']
    
#     # prepare input
#     num_targets = len(target_values)
#     target_values = torch.tensor(target_values,dtype=torch.float32)
#     onehot_input, embedding_input = prepare_dreaming_mof([seed_mof_string],tokenized_info,pad_node=False,noise_level=dream_settings['noise_level'])
#     onehot_input, embedding_input, target_values = onehot_input.to(device), embedding_input.to(device), target_values.to(device)
#     _, seed_smiles = onehot_group_selfies_to_smiles(onehot_input, alphabet, max_len_selfie, group_grammar, symbol_to_idx['[nop]'])
#     seed_node_and_topo = seed_mof_string.split('[.]')[1]

#     # dreaming hyperparameters
#     num_epochs = dream_settings['num_epochs']
#     patience = dream_settings['patience']
#     lr = dream_settings['lr']
#     target_tolerance = dream_settings['target_tolerance']
#     noise_level = dream_settings['noise_level']

#     # tracking variables
#     best_loss = float('inf')
#     no_improvement_counter = 0
#     continuous_targets = []
#     losses = []
#     valid_opt_pathway = []      # valid linker optimisation pathway
#     transmutation_pathway = []  # all molecule transmutation pathway

#     # Training Loop
#     early_stop = False
#     for epoch in range(num_epochs):

#         # backpropogation
#         optimizer = optim.Adam([onehot_input], lr=lr)    # This is placed within the loop to allow us to instantiate onehot_input with different noise
#         optimizer.zero_grad()
#         dreaming_outputs = model(onehot_input, embedding_input)
#         outputs = dreaming_outputs.detach().numpy()[0]
        
#         # calculate penalty on nonlinker molecules
#         mol, perturbed_structure = onehot_group_selfies_to_smiles(onehot_input, alphabet, max_len_selfie, group_grammar, symbol_to_idx['[nop]'])
#         perturbed_group_selfies, _ = onehot_to_selfies(onehot_input, alphabet, max_len_selfie, symbol_to_idx['[nop]'])
#         penalty = connection_penalty(perturbed_structure, penalty_per_connection=dream_settings['penalty_per_connection'])
#         if epoch == 0:
#             predictor_input = prepare_dreaming_edge([perturbed_group_selfies],tokenized_info,noise_level=None)
#             predicted_targets = predictor(predictor_input, embedding_input).detach().numpy()[0]
#             valid_opt_pathway.append({'dreamed_targets': outputs,'predictor_targets':  predicted_targets, 'dreamed_smiles': seed_smiles,'dreamed_selfies': perturbed_group_selfies, 'dreamed_mof_string': perturbed_group_selfies+'[.]'+seed_node_and_topo, 'epoch': epoch})
#             transmutation_pathway.append({'dreamed_targets': outputs, 'predictor_targets':  predicted_targets,'dreamed_smiles': seed_smiles,'dreamed_selfies': perturbed_group_selfies, 'dreamed_mof_string': perturbed_group_selfies+'[.]'+seed_node_and_topo, 'epoch': epoch})
#             opt_flags = ['max' if outputs[i] < target_values[i] else 'min' for i in range(num_targets)]
#             print(f'seed value: {outputs} | target value: {np.array(target_values)} | opt_flag: {opt_flags}\n\n')

#         # add penalty to loss
#         loss = criterion(dreaming_outputs, target_values)
#         total_loss = loss + penalty
#         total_loss.backward()
#         optimizer.step()
        
#         # track variables
#         continuous_targets.append(outputs)
#         losses.append(total_loss.item())
        
#         if perturbed_structure not in [s['dreamed_smiles'] for s in transmutation_pathway]:
#             predictor_input = prepare_dreaming_edge([perturbed_group_selfies],tokenized_info,noise_level=None)
#             predicted_targets = predictor(predictor_input, embedding_input).detach().numpy()[0]
#             transmutation_pathway.append({'dreamed_targets': outputs,'predictor_targets':  predicted_targets,'dreamed_smiles': perturbed_structure,'dreamed_selfies': perturbed_group_selfies, 'dreamed_mof_string': perturbed_group_selfies+'[.]'+seed_node_and_topo, 'epoch': epoch})
#             composition = get_molecule_composition(perturbed_structure)

#             if 'Fr' in composition and composition['Fr'] == 2:
#                 # if first_kpi>valid_opt_pathway[-1]['dreamed_target']:

#                 # Assume opt_flags and predicted_targets are lists of the same length
#                 valid = True
#                 for idx, (flag, predicted_target) in enumerate(zip(opt_flags, predicted_targets)):
#                     # print(flag, predicted_target,valid_opt_pathway[-1]['predictor_targets'], valid_opt_pathway[-1]['predictor_targets'][idx])
#                     if flag == 'max':
#                         if predicted_target <= valid_opt_pathway[-1]['predictor_targets'][idx]:
#                             valid = False
#                             break
#                     elif flag == 'min':
#                         if predicted_target >= valid_opt_pathway[-1]['predictor_targets'][idx]:
#                             valid = False
#                             break
#                 if valid:
#                     if connection_point_graph_distance(perturbed_structure) >= 0.6:
#                         valid_opt_pathway.append({'dreamed_targets': outputs,'predictor_targets':  predicted_targets, 'dreamed_smiles': perturbed_structure,'dreamed_selfies': perturbed_group_selfies, 'dreamed_mof_string': perturbed_group_selfies+'[.]'+seed_node_and_topo, 'epoch': epoch})
#                         print(f'opt flag: {flag}, all targets: {valid_opt_pathway[-1]["predictor_targets"]}, valid linker, valid distance point')
#                     else:
#                         print(f'opt flag: {flag}, all targets: {valid_opt_pathway[-1]["predictor_targets"]}, valid linker, invalid distance point')
#         # terminate dreaming logic
#         if patience != None:
#             if loss.item() < best_loss:
#                 best_loss = loss.item()
#                 no_improvement_counter = 0
#             else:
#                 no_improvement_counter += 1
#                 # if no improvement in loss, but target still not met, get last valid linker and instantiate with new noise

#                 if no_improvement_counter >= patience:
#                     valid=True
#                     for idx, target_value in enumerate(target_values):
#                         if not (((1-target_tolerance)*target_value).numpy() <= valid_opt_pathway[-1]['predictor_targets'][idx] <= ((1+target_tolerance)*target_value).numpy()):
#                             valid = False
#                             break
#                     if not valid:
#                         mol = Chem.MolFromSmiles(valid_opt_pathway[-1]['dreamed_smiles'])
#                         cleaned_selfies = group_grammar.full_encoder(mol)
#                         tokens = sf.split_selfies(cleaned_selfies)
#                         ###################################################################################################
#                         ## TODO: here, using group_grammar to encode mol object leads to new tokens outside of training set.
#                         ##       Need to see if i can restrict the tokens used for encoding.
#                         ###################################################################################################
#                         # if cleaned selfies representation contains tokens outside of training set, then use the unclean selfies instead
#                         if check_token_overlap(tokens, alphabet):
#                             intialise_selfies = cleaned_selfies
#                         else:
#                             intialise_selfies = valid_opt_pathway[-1]['dreamed_selfies']

#                         # cleaned_selfies = sf.encoder(valid_opt_pathway[-1]['dreamed_smiles'])
#                         onehot_input = prepare_dreaming_edge(
#                             [intialise_selfies],
#                             tokenized_info,
#                             noise_level=noise_level
#                             )
#                         no_improvement_counter = 0
#                         patience = 100

#                 # if no improvement in loss but within target range, terminate dreaming
#                 elif no_improvement_counter >= patience:
#                     early_stop = True
#                     print(f"Early stopping triggered. No improvement in MSE loss for {patience} epochs.")
#                     break
#     print('Finished Training')
#     dreaming_losses = {'targets': continuous_targets, 'losses': losses, 'epochs': epoch, 'early_stop': early_stop}
#     return pd.DataFrame(valid_opt_pathway), pd.DataFrame(transmutation_pathway), dreaming_losses

def dream(
        model,
        predictor,
        seed_mof_string: str, 
        target_values: List[float], 
        tokenized_info: dict, 
        group_grammar,
        dream_settings: dict,
        ):
    """
    Generates new MOFs using a dreaming model.

    Args:
        model (nn.Module): The dreaming model used for generating new MOFs.
        predictor (nn.Module): The predictor model used for evaluating the generated MOFs.
        seed_mof_string (str): The seed MOF structure used as a starting point for dreaming.
        target_values List[float]: The (normalised) target values that the dreaming model aims to achieve.
        tokenized_info (dict): Tokenized information used for preparing the input.
        group_grammar: The group grammar used for encoding and decoding the MOF structures.
        dream_settings (dict): Settings for the dreaming process.

    Returns:
        pd.DataFrame: DataFrame containing the valid linker optimization pathway.
        pd.DataFrame: DataFrame containing the molecule transmutation pathway.
        dict: Dictionary containing the targets, losses, epochs, and early stop flag during training.
    """
        
    # model settings
    model.eval() 
    predictor.eval()
    criterion = nn.MSELoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenized info
    alphabet = tokenized_info['alphabet']
    max_len_selfie = tokenized_info['max_len_selfie']
    symbol_to_idx = tokenized_info['symbol_to_idx']
    
    # prepare input
    num_targets = len(target_values)
    target_values = torch.tensor(target_values,dtype=torch.float32)
    onehot_input, embedding_input = prepare_dreaming_mof([seed_mof_string],tokenized_info,pad_node=False,noise_level=dream_settings['noise_level'])
    onehot_input, embedding_input, target_values = onehot_input.to(device), embedding_input.to(device), target_values.to(device)
    _, seed_smiles = onehot_group_selfies_to_smiles(onehot_input, alphabet, max_len_selfie, group_grammar, symbol_to_idx['[nop]'])
    seed_node_and_topo = seed_mof_string.split('[.]')[1]

    # dreaming hyperparameters
    num_epochs = dream_settings['num_epochs']
    patience = dream_settings['patience']
    lr = dream_settings['lr']
    target_tolerance = dream_settings['target_tolerance']
    noise_level = dream_settings['noise_level']

    # tracking variables
    best_loss = float('inf')
    no_improvement_counter = 0
    continuous_targets = []
    losses = []
    valid_opt_pathway = []      # valid linker optimisation pathway
    transmutation_pathway = []  # all molecule transmutation pathway

    # sc_score penalty calculator
    scscorer = SCScorer()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    WEIGHTS_FILE = os.path.join(current_dir, 'scscore_1024uint8_model.ckpt-10654.as_numpy.json.gz')
    scscorer.restore(WEIGHTS_FILE)

    # Training Loop
    early_stop = False
    for epoch in range(num_epochs):

        # backpropogation
        optimizer = optim.Adam([onehot_input], lr=lr)    # This is placed within the loop to allow us to instantiate onehot_input with different noise
        optimizer.zero_grad()
        dreaming_outputs = model(onehot_input, embedding_input)
        outputs = dreaming_outputs.detach().numpy()[0]

        # calculate penalty on nonlinker molecules
        mol, perturbed_structure = onehot_group_selfies_to_smiles(onehot_input, alphabet, max_len_selfie, group_grammar, symbol_to_idx['[nop]'])
        perturbed_group_selfies, _ = onehot_to_selfies(onehot_input, alphabet, max_len_selfie, symbol_to_idx['[nop]'])
        penalty = connection_penalty(perturbed_structure, penalty_per_connection=dream_settings['penalty_per_connection'])
        sc_penalty = sc_score_penalty(perturbed_structure, scscorer)
        
        if epoch == 0:
            predictor_input = prepare_dreaming_edge([perturbed_group_selfies],tokenized_info,noise_level=None)
            predicted_targets = predictor(predictor_input, embedding_input).detach().numpy()[0]
            valid_opt_pathway.append({'dreamed_targets': outputs,'predictor_targets':  predicted_targets, 'dreamed_smiles': seed_smiles,'dreamed_selfies': perturbed_group_selfies, 'dreamed_mof_string': perturbed_group_selfies+'[.]'+seed_node_and_topo, 'epoch': epoch})
            transmutation_pathway.append({'dreamed_targets': outputs, 'predictor_targets':  predicted_targets,'dreamed_smiles': seed_smiles,'dreamed_selfies': perturbed_group_selfies, 'dreamed_mof_string': perturbed_group_selfies+'[.]'+seed_node_and_topo, 'epoch': epoch})
            opt_flags = ['max' if outputs[i] < target_values[i] else 'min' for i in range(num_targets)]
            print(f'seed value: {outputs} | target value: {np.array(target_values)} | opt_flag: {opt_flags}\n\n')

        # add penalty to loss
        loss = criterion(dreaming_outputs, target_values)
        total_loss = loss + penalty
        total_loss.backward()
        optimizer.step()
        
        # track variables
        continuous_targets.append(outputs)
        losses.append(total_loss.item())
        
        if perturbed_structure not in [s['dreamed_smiles'] for s in transmutation_pathway]:
            predictor_input = prepare_dreaming_edge([perturbed_group_selfies],tokenized_info,noise_level=None)
            predicted_targets = predictor(predictor_input, embedding_input).detach().numpy()[0]
            transmutation_pathway.append({'dreamed_targets': outputs,'predictor_targets':  predicted_targets,'dreamed_smiles': perturbed_structure,'dreamed_selfies': perturbed_group_selfies, 'dreamed_mof_string': perturbed_group_selfies+'[.]'+seed_node_and_topo, 'epoch': epoch})
            composition = get_molecule_composition(perturbed_structure)

            if 'Fr' in composition and composition['Fr'] == 2:
                # if first_kpi>valid_opt_pathway[-1]['dreamed_target']:

                # Assume opt_flags and predicted_targets are lists of the same length
                valid = True
                for idx, (flag, predicted_target) in enumerate(zip(opt_flags, predicted_targets)):
                    # print(flag, predicted_target,valid_opt_pathway[-1]['predictor_targets'], valid_opt_pathway[-1]['predictor_targets'][idx])
                    if flag == 'max':
                        if predicted_target <= valid_opt_pathway[-1]['predictor_targets'][idx]:
                            valid = False
                            break
                    elif flag == 'min':
                        if predicted_target >= valid_opt_pathway[-1]['predictor_targets'][idx]:
                            valid = False
                            break
                if valid:
                    if connection_point_graph_distance(perturbed_structure) >= 0.6:
                        # ************************** SC PENALTY CONSTRAINT ************************
                        if sc_penalty <= 400:
                            valid_opt_pathway.append({'dreamed_targets': outputs,'predictor_targets':  predicted_targets, 'dreamed_smiles': perturbed_structure,'dreamed_selfies': perturbed_group_selfies, 'dreamed_mof_string': perturbed_group_selfies+'[.]'+seed_node_and_topo, 'epoch': epoch})
                            print(f'opt flag: {flag}, all targets: {valid_opt_pathway[-1]["predictor_targets"]}, valid linker, valid distance point')
                    else:
                        print(f'opt flag: {flag}, all targets: {valid_opt_pathway[-1]["predictor_targets"]}, valid linker, invalid distance point')
        # terminate dreaming logic
        if patience != None:
            if loss.item() < best_loss:
                best_loss = loss.item()
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1
                # if no improvement in loss, but target still not met, get last valid linker and instantiate with new noise

                if no_improvement_counter >= patience:
                    valid=True
                    for idx, target_value in enumerate(target_values):
                        if not (((1-target_tolerance)*target_value).numpy() <= valid_opt_pathway[-1]['predictor_targets'][idx] <= ((1+target_tolerance)*target_value).numpy()):
                            valid = False
                            break
                    if not valid:
                        mol = Chem.MolFromSmiles(valid_opt_pathway[-1]['dreamed_smiles'])
                        cleaned_selfies = group_grammar.full_encoder(mol)
                        tokens = sf.split_selfies(cleaned_selfies)
                        ###################################################################################################
                        ## TODO: here, using group_grammar to encode mol object leads to new tokens outside of training set.
                        ##       Need to see if i can restrict the tokens used for encoding.
                        ###################################################################################################
                        # if cleaned selfies representation contains tokens outside of training set, then use the unclean selfies instead
                        if check_token_overlap(tokens, alphabet):
                            intialise_selfies = cleaned_selfies
                        else:
                            intialise_selfies = valid_opt_pathway[-1]['dreamed_selfies']

                        # cleaned_selfies = sf.encoder(valid_opt_pathway[-1]['dreamed_smiles'])
                        onehot_input = prepare_dreaming_edge(
                            [intialise_selfies],
                            tokenized_info,
                            noise_level=noise_level
                            )
                        no_improvement_counter = 0
                        patience = 100

                # if no improvement in loss but within target range, terminate dreaming
                elif no_improvement_counter >= patience:
                    early_stop = True
                    print(f"Early stopping triggered. No improvement in MSE loss for {patience} epochs.")
                    break
    print('Finished Training')
    dreaming_losses = {'targets': continuous_targets, 'losses': losses, 'epochs': epoch, 'early_stop': early_stop}
    return pd.DataFrame(valid_opt_pathway), pd.DataFrame(transmutation_pathway), dreaming_losses


def run_dream_exp(
    dreaming_model, 
    predictor_model, 
    seed_mof_string, 
    target_values, 
    tokenized_info, 
    group_grammar, 
    dream_settings,
    iterations=5):

    n = []
    seed_mof_string_to_opt = seed_mof_string
    for _ in range(iterations):
        intermediate_valid_opt_pathway, _, _= dream(
            dreaming_model,
            predictor_model,
            seed_mof_string_to_opt,
            target_values,
            tokenized_info,
            group_grammar, 
            dream_settings
            )
        seed_mof_string_to_opt = intermediate_valid_opt_pathway.iloc[-1]['dreamed_mof_string']
        n.append(intermediate_valid_opt_pathway)
        mol = Chem.MolFromSmiles(intermediate_valid_opt_pathway.iloc[-1]['dreamed_smiles'])
        cleaned_selfies = group_grammar.full_encoder(mol)
        tokens = sf.split_selfies(cleaned_selfies)

        # if cleaned selfies representation contains tokens outside of training set, then use the unclean selfies instead
        if check_token_overlap(tokens, tokenized_info['alphabet']):
            seed_mof_string_to_opt = cleaned_selfies+'[.]'+seed_mof_string.split('[.]')[1]
        else:
            seed_mof_string_to_opt = intermediate_valid_opt_pathway.iloc[-1]['dreamed_mof_string']
    valid_opt_pathway = pd.concat(n)
    valid_opt_pathway.reset_index(inplace=True,drop=True)

    return valid_opt_pathway


def predict_kpi(predictor, seed_mof_string: str, tokenized_info: dict):
    # model settings
    predictor.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # tokenized info
    alphabet = tokenized_info['alphabet']
    max_len_selfie = tokenized_info['max_len_selfie']
    symbol_to_idx = tokenized_info['symbol_to_idx']
    
    # prepare input 
    onehot_input, embedding_input = prepare_dreaming_mof([seed_mof_string],tokenized_info,pad_node=False,noise_level=None)
    onehot_input, embedding_input = onehot_input.to(device), embedding_input.to(device)
    predicted_targets, attn_weights1, attn_weights2 = predictor(onehot_input, embedding_input, return_attention_weights=True)
    return predicted_targets, attn_weights1, attn_weights2