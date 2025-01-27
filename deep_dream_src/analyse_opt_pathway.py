import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors, DataStructs, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
from io import BytesIO
from utils import compute_fingerprints, get_molecule_composition
import statistics
from collections import defaultdict
import selfies as sf
from dreaming_functions import predict_kpi, connection_point_graph_distance

def normalise_key(key):
    if key.startswith('[:#'):
        return '['+key[4:]
    elif key.startswith('[:='):
        return '['+key[4:]
    elif key.startswith('[:'):
        return '['+key[3:]
    else:
        return key

def analyse_attn_changes(df, seed_mof_string, predictor_model, tokenized_info):
    changes = []
    for i in range(1, len(df)):
        previous_sf = df['dreamed_selfies'].iloc[i-1]
        current_sf = df['dreamed_selfies'].iloc[i]

        # get MOF strings
        seed_mof_node_and_topo = seed_mof_string.split('[.]')[1]
        previous_mof_string = f'{previous_sf}[.]{seed_mof_node_and_topo}'
        current_mof_string = f'{current_sf}[.]{seed_mof_node_and_topo}'

        # get attention weights
        _, w1_prev, _ = predict_kpi(predictor_model,previous_mof_string,tokenized_info)
        _, w1_curr, _ = predict_kpi(predictor_model,current_mof_string,tokenized_info)

        # get tokens
        tokens_prev = list(sf.split_selfies(previous_sf))
        tokens_curr = list(sf.split_selfies(current_sf))

        # get normalized attention weights
        len_edge_selfies_prev = sf.len_selfies(previous_sf)
        len_edge_selfies_curr = sf.len_selfies(current_sf)
        attn_edge_prev = w1_prev.detach().numpy().flatten()[:len_edge_selfies_prev+1] / np.sum(w1_prev.detach().numpy().flatten()[:len_edge_selfies_prev+1])
        attn_edge_curr = w1_curr.detach().numpy().flatten()[:len_edge_selfies_curr+1] / np.sum(w1_curr.detach().numpy().flatten()[:len_edge_selfies_curr+1])

        # print(f'Previous: {attn_edge_prev}')
        # print(f'Current: {attn_edge_curr}')
        # print(f'Tokens: {tokens_prev} -> {tokens_curr}')
        # Append results to the changes list
        changes.append({
            'step': i,
            'token_changes': [tokens_prev, tokens_curr],
            'attn_changes': [attn_edge_prev, attn_edge_curr]
        })
        return pd.DataFrame(changes)
    

def attn_stats(df: pd.DataFrame):

    token2weights = defaultdict(list)

    # stored in the form of [previous_tokens, current_tokens]
    all_tokens = [x[0] for x in df['token_changes']] + [df['token_changes'].iloc[-1][1]]
    all_attn_weights = [x[0] for x in df['attn_changes']] + [df['attn_changes'].iloc[-1][1]]

    # Iterate over each pair of (tokens, weights)
    for tokens_list, weights_array in zip(all_tokens, all_attn_weights):
        # Zip them together so each token matches its weight
        for token, weight in zip(tokens_list, weights_array):
            token = normalise_key(token) # group non-unique Group tokens
            token2weights[token].append(weight)

    stats_list = []
    for token, weights in token2weights.items():
        mean_val = statistics.mean(weights)
        median_val = statistics.median(weights)
        # watch out for "no unique mode" errors here
        try:
            mode_val = statistics.mode(weights)
        except statistics.StatisticsError:
            mode_val = None  # or handle multiple modes
        stats_list.append((token, mean_val, median_val, mode_val))

    df_stats = pd.DataFrame(stats_list, columns=["token", "mean", "median", "mode"])
    df_stats = df_stats.sort_values("median", ascending=False).reset_index(drop=True)
    return df_stats

def analyse_token_changes(df):
    changes = []
    for i in range(1, len(df)):
        previous_tokens = list(sf.split_selfies(df['dreamed_selfies'].iloc[i-1]))
        current_tokens = list(sf.split_selfies(df['dreamed_selfies'].iloc[i]))

        # Initialise placeholders for additions, removals, and substitutions
        substitutions, additions, removals = [], [], []
        max_len = max(len(previous_tokens), len(current_tokens))

        # Process token lists for substitutions and identifying additions/removals
        for j in range(max_len):
            if j < len(previous_tokens) and j < len(current_tokens):
                if previous_tokens[j] != current_tokens[j]:
                    substitutions.append(f'{previous_tokens[j]}->{current_tokens[j]}')
            elif j >= len(previous_tokens):
                additions.append(current_tokens[j])
            elif j >= len(current_tokens):
                removals.append(previous_tokens[j])

        # Append results to the changes list
        changes.append({
            'step': i,
            'substitutions': substitutions,
            'additions': additions,
            'removals': removals
        })

    return pd.DataFrame(changes)


def analyse_composition_change(prev_smiles, curr_smiles):
    mol_prev = Chem.MolFromSmiles(prev_smiles)
    mol_curr = Chem.MolFromSmiles(curr_smiles)

    formula_prev = get_molecule_composition(prev_smiles)
    formula_curr = get_molecule_composition(curr_smiles)

    return {
        'previous_formula': formula_prev,
        'new_formula': formula_curr,
        'composition_change': f'{rdMolDescriptors.CalcMolFormula(mol_prev)}->{rdMolDescriptors.CalcMolFormula(mol_curr)}'
    }


def analyse_structural_change(prev_smiles, curr_smiles):
    mol_prev = Chem.MolFromSmiles(prev_smiles)
    mol_curr = Chem.MolFromSmiles(curr_smiles)

    # Compare bond differences
    bonds_prev = set((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol_prev.GetBonds())
    bonds_curr = set((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol_curr.GetBonds())

    added_bonds = bonds_curr - bonds_prev
    removed_bonds = bonds_prev - bonds_curr

    return {
        'added_bonds': added_bonds,
        'removed_bonds': removed_bonds
    }


def calculate_similarity(prev_smiles, curr_smiles):
    
    fps = compute_fingerprints([prev_smiles, curr_smiles])
    # Calculate Tanimoto similarity
    similarity = DataStructs.TanimotoSimilarity(fps[0], fps[1])

    return {
        'tanimoto_similarity': similarity
    }


def compute_smiles_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    properties = {
        'MolecularWeight': Descriptors.MolWt(mol),
        'NumRings': Descriptors.RingCount(mol),
        'NumHBD': Descriptors.NumHDonors(mol),
        'NumHBA': Descriptors.NumHAcceptors(mol),
        'Polarizability': Descriptors.TPSA(mol),  # Topological Polar Surface Area
        'LogP': Descriptors.MolLogP(mol),
    }
    return properties


def connection_point_atomic_distance(smiles,connection='Fr'):
    molecule = Chem.MolFromSmiles(smiles)
    # Add hydrogens (if necessary) and generate a 3D conformation
    molecule = Chem.AddHs(molecule)

    # Identify the placeholder atoms by their atomic number
    placeholder_atomic_number = Chem.Atom(connection).GetAtomicNum() 
    placeholder_atom_indices = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetAtomicNum() == placeholder_atomic_number]

    # Ensure that there are exactly two placeholder atoms
    if len(placeholder_atom_indices) == 2:
        AllChem.EmbedMolecule(molecule, AllChem.ETKDG())
        AllChem.UFFOptimizeMolecule(molecule)
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
    else:
        return None, None


def split_dataframe_on_valid_transmutation(df):
    sub_dataframes = []
    start_idx = 0

    for idx, value in enumerate(df['valid_transmutation']):
        if value == 1:
            sub_dataframes.append(df.iloc[start_idx:idx+1].reset_index(drop=True))
            start_idx = idx + 1

    return sub_dataframes


def analyse_transmutation_pathway(transmutation_pathway, valid_opt_pathway, seed_mof_string, predictor_model, tokenized_info):
    results = []
    for i in range(1, len(transmutation_pathway)):
        prev_smiles = transmutation_pathway['dreamed_smiles'].iloc[i-1]
        curr_smiles = transmutation_pathway['dreamed_smiles'].iloc[i]
        # print(f'prev_smiles: {prev_smiles}, curr_smiles: {curr_smiles}')
        
        # Perform all analyses
        composition_change = analyse_composition_change(prev_smiles, curr_smiles)
        structural_change = analyse_structural_change(prev_smiles, curr_smiles)
        similarity = calculate_similarity(prev_smiles, curr_smiles)

        # SELFIES changes
        selfies_changes = analyse_token_changes(transmutation_pathway.iloc[[i-1, i]])
        attn_changes = analyse_attn_changes(transmutation_pathway.iloc[[i-1, i]], seed_mof_string, predictor_model, tokenized_info)

        # SMILES properties
        prev_properties = compute_smiles_properties(prev_smiles)
        curr_properties = compute_smiles_properties(curr_smiles)

        # Property changes
        property_diff = {key: curr_properties[key] - prev_properties[key] for key in prev_properties.keys()}
        property_changes = {key: f'{curr_properties[key]:.3f}->{prev_properties[key]:.3f}' for key in prev_properties.keys()}

        results.append({
            'transmutation': i,
            'selfies_changes': f'{transmutation_pathway["dreamed_selfies"].iloc[i-1]} -> {transmutation_pathway["dreamed_selfies"].iloc[i]}',
            'token_substitutions': selfies_changes['substitutions'][0],
            'token_additions': selfies_changes['additions'][0],
            'token_removals': selfies_changes['removals'][0],
            'token_changes': attn_changes['token_changes'][0],
            'attn_changes': attn_changes['attn_changes'][0],
            'smiles_changes': f'{prev_smiles}->{curr_smiles}',
            'property_changes': property_changes,
            'property_diff': property_diff,
            **composition_change,
            **structural_change,
            **similarity,
            # 'connection_point_atomic_dist': f'{connection_point_atomic_distance(prev_smiles)[1]}->{connection_point_atomic_distance(curr_smiles)[1]}',
            'connection_point_graph_distance': f'{connection_point_graph_distance(prev_smiles)}-> {connection_point_graph_distance(curr_smiles)}'
        })
    results = pd.DataFrame(results)

    # identify valid_opt_pathway transmutations
    overlap_mask = transmutation_pathway['dreamed_selfies'].isin(valid_opt_pathway['dreamed_selfies'])
    overlap_indices = transmutation_pathway[overlap_mask].index-1
    valid_transmutation_flag = np.zeros(transmutation_pathway.shape[0]-1, dtype=int)
    valid_transmutation_flag[overlap_indices[1:]] = 1
    results['valid_transmutation'] = valid_transmutation_flag
    return results


def mol_to_image(
    Sm, 
    size=(400, 400), 
    line_width=3.0,
    padding=0.0,      
    do_crop=True      
    ):
    
    AllChem.Compute2DCoords(Sm)
    drawer = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = drawer.drawOptions()
    opts.bondLineWidth = int(line_width)
    
    # Set the padding around the molecule
    opts.padding = padding
    
    # Draw
    drawer.DrawMolecule(Sm)
    drawer.FinishDrawing()
    
    # Convert to PIL Image
    img = Image.open(BytesIO(drawer.GetDrawingText()))
    
    if do_crop:
        # Crop out all the blank space
        img = img.crop(img.getbbox())
    
    return img

def plot_valid_transmutation_pathway(
        transmutation_pathway,
        valid_opt_pathway, 
        seed_mof_string, 
        predictor_model, 
        predictor_scaler,
        tokenized_info, 
        target_log=False, 
        width=4,
        pad=-4.5,
        column_height=1.75,
        save_fig=None
        ):

    # Filter for valid transmutations
    df = analyse_transmutation_pathway(transmutation_pathway, valid_opt_pathway,seed_mof_string,predictor_model,tokenized_info)
    valid_df = df[df['valid_transmutation'] == 1].reset_index(drop=True)

    # split the dataframe into sub-dataframes based on valid transmutations
    split_dataframes = split_dataframe_on_valid_transmutation(df)

    # Calculate number of molecules and rows
    num_molecules = len(valid_df)+1  # +1 for the initial molecule
    num_rows = (num_molecules + width - 1) // width  # Ceiling division

    height_ratios = []
    for i in range(num_rows * 2 - 1):
        if i % 2 == 0:
            # This is a "molecule row"
            height_ratios.append(1.0)
        else:
            # This is an "arrow row"
            height_ratios.append(column_height)  # or 0.6, 0.8, 1.5, etc.


    # # If arrow_texts is None or too short, pad with None
    # if arrow_texts is None:
    #     arrow_texts = [None]*num_molecules
    # else:
    #     # Ensure arrow_texts is at least long enough so we can index safely
    #     if len(arrow_texts) < num_molecules:
    #         arrow_texts += [None]*(num_molecules - len(arrow_texts))

    # Create figure with extra subplots for arrows
    fig, ax = plt.subplots(
        nrows=num_rows * 2 - 1, 
        ncols=width * 2 - 1, 
        figsize=(width * 3, num_rows * 2),
        gridspec_kw={'height_ratios': height_ratios}
    )
    ax = ax.flatten()

    # Turn off all axes initially
    for ax_obj in ax:
        ax_obj.axis('off')

    # Helper for flattened indexing
    def subplot_index(row, col):
        return row * (width * 2 - 1) + col

    # Plot the initial molecule
    initial_smiles = valid_opt_pathway['dreamed_smiles'].iloc[0]
    initial_mol = Chem.MolFromSmiles(initial_smiles)
    # initial_img = Draw.MolToImage(initial_mol, size=(200, 200))
    img = mol_to_image(initial_mol, size=(400, 400), padding=0.0,line_width=3,do_crop=False if num_rows == 1 else True)
    ax[0].imshow(img)
    
    # print(compute_statistics(split_dataframes[0], top_n=5))
    for iter, row in valid_df.iterrows():
        smiles = row['smiles_changes'].split('->')[1]
        mol = Chem.MolFromSmiles(smiles)
        # img = Draw.MolToImage(mol, size=(200, 200),fitImage=True)
        img = mol_to_image(mol, size=(400, 400), padding=0.0,line_width=3,do_crop=False if num_rows == 1 else True)
        i = iter+1
        # Determine the expanded row/col in the snake layout
        molecule_row = (i // width) * 2
        if (i // width) % 2 == 0:
            molecule_col = (i % width) * 2
        else:
            molecule_col = (width - 1 - (i % width)) * 2

        # Show the molecule
        ax_idx = subplot_index(molecule_row, molecule_col)
        ax[ax_idx].imshow(img)
        # ax[ax_idx].axis('off')

    def draw_arrow(arrow_ax, direction='lr'):
        """
        direction: 'lr' (left->right), 'rl' (right->left), 'down'
        """
        arrow_ax.set_xlim(0, 1)
        arrow_ax.set_ylim(0, 1)
        arrow_ax.set_xticks([])
        arrow_ax.set_yticks([])

        if direction == 'lr':
            # Left->Right arrow
            arrow_ax.arrow(
                0.0, 0.52,  # start near the left, a bit higher up
                0.95, 0.0,  # dx=0.6, dy=0
                head_width=0.04, length_includes_head=True,
                fc='grey', ec='grey'
            )
        elif direction == 'rl':
            # Right->Left arrow
            arrow_ax.arrow(
                1.0, 0.52,  # start near the right
                -0.95, 0.0,  # dx=-0.6, dy=0
                head_width=0.04, length_includes_head=True,
                fc='grey', ec='grey'
            )
        elif direction == 'down':
            # Down arrow
            arrow_ax.arrow(
                0.45, 0.92,  # start up higher
                0.0, -0.75,  # dx=0, dy=-0.5
                head_width=0.04, length_includes_head=True,
                fc='grey', ec='grey'
            )

    def add_inset_axes_below(parent_ax, direction='lr'):
        """
        If the arrow is left/right, the mini-plot goes BELOW it.
        If the arrow is down, the mini-plot goes to the LEFT of it.
        """
        if direction in ('lr', 'rl'):
            # For horizontal arrows, put the mini-plot in bottom portion
            # e.g. y from 0.05 to 0.45
            data_ax = parent_ax.inset_axes([0.25, 0.0, 0.55, 0.45])
        else:
            # direction == 'down'
            # Put the mini-plot on the left
            # e.g. x from 0.05 to 0.45
            data_ax = parent_ax.inset_axes([0, 0.4, 0.4, 0.35])

        return data_ax

    def add_inset_axes_above(parent_ax, direction='lr'):
        """
        For left/right arrows, place the mini-plot in the top portion.
        For down arrow, place on the right portion.
        """
        if direction in ('lr', 'rl'):
            # top portion
            data_ax = parent_ax.inset_axes([0.35, 0.72, 0.55, 0.65])
        else:
            # direction == 'down' => right portion
            data_ax = parent_ax.inset_axes([0.55, 0.55, 0.45, 0.35])
        return data_ax
    
    def plot_target_data(inset_ax, df, iter):
        targets = df['predictor_targets'].apply(
            lambda x: predictor_scaler.inverse_transform(
                np.array(x).reshape(1, -1)
                ).item()
        )
        inset_ax.plot(
            np.arange(df.iloc[:iter].shape[0]),
            targets.iloc[:iter],
            color='red',
            marker='o',        
            linestyle='-',     
            markersize=5      
        )
        inset_ax.set_xlabel("Step", fontsize=8)
        inset_ax.set_ylabel("Target", fontsize=8)
        inset_ax.set_ylim(np.min(targets)*0.95, np.max(targets)*1.05)
        inset_ax.set_xlim(-0.5, df.shape[0]-0.5)
        inset_ax.spines['top'].set_visible(False)
        inset_ax.spines['right'].set_visible(False)
        if target_log:
            inset_ax.set_yscale('log')
        # Make the tick labels smaller so it fits better
        inset_ax.tick_params(labelsize=8)

    def plot_attn_data(inset_ax,attn_stat_df,orientation='horizontal'):
        helper_alphabet = ['[nop]','[#Branch]','[->]', '[=Ring1]', '[=Ring2]','[Branch]','[=Branch]', '[Ring1]', '[Ring2]','[pop]', '[FrH0]']
        attn_without_helper = attn_stat_df[~attn_stat_df['token'].isin(helper_alphabet)]
        attn_to_plot = attn_without_helper.head(4)['median'].tolist()+ [0] + attn_without_helper.tail(2)['median'].tolist()
        tokens_to_plot = attn_without_helper.head(4)['token'].tolist()+ ['⋮']+ attn_without_helper.tail(2)['token'].tolist()

        if orientation == 'horizontal':
            x_positions = range(len(attn_to_plot))
            inset_ax.bar(x_positions,attn_to_plot,color='steelblue',alpha=0.75)
            inset_ax.set_ylabel("Attention weight",fontsize=8)
            inset_ax.set_yticks([])
            inset_ax.set_xticks(x_positions)
            inset_ax.set_xticklabels(tokens_to_plot,rotation=90)
            inset_ax.spines['top'].set_visible(False)
            inset_ax.spines['right'].set_visible(False)
            inset_ax.tick_params(labelsize=8)

        if orientation == 'vertical':
            y_positions = range(len(attn_to_plot))
            inset_ax.barh(y_positions,attn_to_plot,color='steelblue',alpha=0.75)
            inset_ax.set_xlabel("Attention weight",fontsize=8)
            inset_ax.set_yticks(y_positions)
            inset_ax.set_yticklabels(tokens_to_plot)
            inset_ax.spines['top'].set_visible(False)
            inset_ax.spines['right'].set_visible(False)
            inset_ax.set_xticks([])
            inset_ax.tick_params(labelsize=8)

    
    # Add arrows + data plots in the "in-between" subplots 
    for i in range(num_molecules-1):
        # print(i)
        row_num = i // width
        row_expanded = row_num * 2

        # Snake logic for columns
        if row_num % 2 == 0:
            col_expanded = (i % width) * 2
        else:
            col_expanded = (width - 1 - (i % width)) * 2

        # get attention stats 
        attn_stat_df = attn_stats(split_dataframes[i])

        # Check if there's a next molecule in the same row
        if (i + 1 < num_molecules) and ((i + 1) // width == row_num):
            # Plot left->right or right->left arrow
            arrow_ax_idx = (
                subplot_index(row_expanded, col_expanded + 1) 
                if row_num % 2 == 0 
                else subplot_index(row_expanded, col_expanded - 1)
            )
            arrow_main_ax = ax[arrow_ax_idx]
            arrow_main_ax.axis('off')

            direction = 'lr' if row_num % 2 == 0 else 'rl'
            draw_arrow(arrow_main_ax, direction=direction)

            # Add optional text above the arrow
            # txt = 'TEXT'
            # if txt is not None:
            #     # Coordinates near the top center if arrow is horizontal
            #     arrow_main_ax.text(
            #         0.5, 0.9, txt,
            #         ha='center', va='bottom', fontsize=8, color='blue'
            #     )

            # Put the mini-plot below the arrow
            data_ax = add_inset_axes_below(arrow_main_ax, direction=direction)
            plot_target_data(data_ax,valid_opt_pathway,i+2)

            # plot attention data above the arrow
            data_ax_above = add_inset_axes_above(arrow_main_ax, direction=direction)
            plot_attn_data(data_ax_above, attn_stat_df,orientation='vertical')
        else:
            # Last molecule in the row or the final molecule
            if row_num < num_rows - 1:
                # We have a next row, so draw a downward arrow
                down_ax_idx = subplot_index(row_expanded + 1, col_expanded)
                down_main_ax = ax[down_ax_idx]
                down_main_ax.axis('off')

                draw_arrow(down_main_ax, direction='down')

                # Add optional text above the down arrow
                # txt = 'TEXT'
                # if txt is not None:
                #     # For the down arrow, place text near the top, but shift horizontally
                #     down_main_ax.text(
                #         0.7, 0.9, txt,
                #         ha='center', va='bottom', fontsize=8, color='blue'
                #     )
                # Put the mini-plot to the left of the arrow
                data_ax = add_inset_axes_below(down_main_ax, direction='down')
                plot_target_data(data_ax, valid_opt_pathway,i+2)

                data_ax_above = add_inset_axes_above(down_main_ax, direction='down')
                plot_attn_data(data_ax_above, attn_stat_df,orientation='horizontal')

    # A small negative pad sometimes helps squeeze subplots if you see spacing
    plt.tight_layout(pad=pad)
    plt.show()

    if isinstance(save_fig, str):
        fig.savefig(save_fig, dpi=300, bbox_inches='tight')

def compute_statistics(change_df, top_n=5):

    # Total changes
    total_substitutions = sum(len(row['token_substitutions']) for index, row in change_df.iterrows())
    total_additions = sum(len(row['token_additions']) for index, row in change_df.iterrows())
    total_removals = sum(len(row['token_removals']) for index, row in change_df.iterrows())

    # Average changes per step
    avg_subs = total_substitutions / len(change_df)
    avg_adds = total_additions / len(change_df)
    avg_rems = total_removals / len(change_df)

    # Most common tokens involved in changes
    all_additions = [token for index, row in change_df.iterrows() for token in row['token_additions']]
    all_removals = [token for index, row in change_df.iterrows() for token in row['token_removals']]
    all_substitutions = [sub for index, row in change_df.iterrows() for sub in row['token_substitutions']]
    # all_additions_plus_substitutions = [sub.split('->')[1] for sub in all_substitutions]+ all_additions
    # all_removals_plus_substitutions =  [sub.split('->')[0] for sub in all_substitutions] + all_removals

    from collections import Counter
    most_common_adds = Counter(all_additions).most_common()
    most_common_rems = Counter(all_removals).most_common()
    most_common_subs = Counter(all_substitutions).most_common()
    # most_common_adds_plus_subs = Counter(all_additions_plus_substitutions).most_common()
    # most_common_rems_plus_subs = Counter(all_removals_plus_substitutions).most_common()

    return {
        'total_substitutions': total_substitutions,
        'total_additions': total_additions,
        'total_removals': total_removals,
        'average_substitutions_per_mutation': avg_subs,
        'average_additions_per_mutation': avg_adds,
        'average_removals_per_mutation': avg_rems,
        'most_common_additions': dict(most_common_adds[:top_n]),
        'most_common_removals': dict(most_common_rems[:top_n]),
        'most_common_substitutions': dict(most_common_subs[:top_n]),
        # 'most_common_additions_plus_substitutions': dict(most_common_adds_plus_subs[:top_n]),
        # 'most_common_removals_plus_substitutions': dict(most_common_rems_plus_subs[:top_n])
    }