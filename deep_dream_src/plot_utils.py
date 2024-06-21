import os
import selfies as sf
from rdkit.Chem import MolFromSmiles, Draw
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from io import BytesIO
from PIL import Image
from rdkit import Chem
from ase.io import read 
from ase.visualize.plot import plot_atoms
from sklearn.metrics import r2_score, mean_absolute_error
import glob
from scipy.stats import gaussian_kde, spearmanr
import torch 

constraints = {'H': 1, 'F': 1, 'Cl': 1, 'Br': 1, 'I': 1, 'B': 3, 'B+1': 2, 'B-1': 4, 'O': 2, 'O+1': 3, 'O-1': 1, 'N': 3, 'N+1': 4, 'N-1': 2, 'C': 4, 'C+1': 5, 'C-1': 3, 'P': 5,
               'P+1': 6, 'P-1': 4, 'S': 6, 'S+1': 7, 'S-1': 5, '?': 8, 'Fr': 1}
sf.set_semantic_constraints(constraints)


def mol_with_atom_index(mol):
    mol_c = Chem.Mol(mol)
    for atom in mol_c.GetAtoms():
        atom.SetProp("atomNote", str(atom.GetIdx()))
    return mol_c

# def draw_atom_idx(Sm, size=(200, 200), annotate_index=False, line_thickness=1.0):
#     if annotate_index:
#         Sm = mol_with_atom_index(Sm)

#     AllChem.Compute2DCoords(Sm)
#     X = rdMolDraw2D.MolDraw2DCairo(*size, bondLineWidth=line_thickness)
#     X.DrawMolecule(Sm)
#     X.FinishDrawing()
#     return Image.open(BytesIO(X.GetDrawingText()))

def draw_atom_idx(Sm, size=(200, 200), annotate_index=False, line_width=1.0):
    if annotate_index:
        Sm = mol_with_atom_index(Sm)

    AllChem.Compute2DCoords(Sm)
    X = rdMolDraw2D.MolDraw2DCairo(*size)
    opts = X.drawOptions()  # Get the DrawOptions object
    opts.bondLineWidth = int(line_width)  # Set the bond line width
    
    X.DrawMolecule(Sm)
    X.FinishDrawing()
    return Image.open(BytesIO(X.GetDrawingText()))

def draw_smiles_linker(smiles: list, molsPerRow=3, subImgSize=(200, 200),line_width=1.0):
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow: nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    full_image = Image.new('RGBA', fullSize )
    for ii, mol in enumerate(mols):
        column = ii % molsPerRow
        row = ii // molsPerRow
        offset = ( column*subImgSize[0], row * subImgSize[1] )
        sub = draw_atom_idx(mol, size=subImgSize,line_width=line_width)
        full_image.paste(sub, box=offset)
    return full_image

def draw_selfies_linker(selfies: list, molsPerRow=3, subImgSize=(200, 200)):
    mols = [Chem.MolFromSmiles(sf.decoder(selfie)) for selfie in selfies]
    nRows = len(mols) // molsPerRow
    if len(mols) % molsPerRow: nRows += 1
    fullSize = (molsPerRow * subImgSize[0], nRows * subImgSize[1])
    full_image = Image.new('RGBA', fullSize )
    for ii, mol in enumerate(mols):
        column = ii % molsPerRow
        row = ii // molsPerRow
        offset = ( column*subImgSize[0], row * subImgSize[1] )
        sub = draw_atom_idx(mol, size=subImgSize)
        full_image.paste(sub, box=offset)
    return full_image


def view_ase_mofs(path_to_cifs: str): 
    '''
    Visualise cif files contained within path_to_cifs directory
    '''
    unsorted_files = glob.glob(os.path.join(path_to_cifs,'*.cif'))
    files = sorted(unsorted_files, key=lambda name: int(name.split('transformation')[1].split('.cif')[0]))

    if len(files) <= 3:
        fig, axes = plt.subplots(1,len(files), figsize=(16,5))
        for i in range(len(files)):
            ax = axes[i]
            atoms = read(files[i])
            plot_atoms(atoms, ax=ax)
            ax.set_title(files[i].split("\\")[-1].split(".")[0], fontsize=10)
            ax.set_axis_off()
        plt.show()
    else:
        rows = int(np.ceil(len(files)/ 3))
        fig, axes = plt.subplots(rows, 3, figsize=(16, 5*rows))
        for i in range(len(files)):
            ax = axes[i//3, i%3]
            atoms = read(files[i])
            plot_atoms(atoms, ax=ax)
            ax.set_title(files[i].split("\\")[-1].split(".")[0], fontsize=10)
            ax.set_axis_off()


def parity_plot(y_test: np.array, y_pred: np.array, ax, scale='linear'):
    """Plots the predictions of a model against the true values.
    
    Args:
        y_test (numpy.ndarray): The true values of the test set.
        y_pred (numpy.ndarray): The predicted values of the test set.

    Returns:
        None
    """
    # Calculate the point density using a Gaussian kernel density estimation
    xy = np.vstack([y_test, y_pred])
    density = gaussian_kde(xy)(xy)
    ax.scatter(y_test, y_pred,s=10,c=density,cmap='viridis')
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k-', lw=1)
    ax.set(
        xlabel='Actual',
        ylabel='Predicted',
        title=f'R2: {r2_score(y_test, y_pred):.2f}, SRCC: {spearmanr(y_test,y_pred)[0]:.2f}, MAE: {mean_absolute_error(y_test,y_pred):.3f}'
        )
    ax.set_xscale(scale)
    ax.set_yscale(scale)
    ax.set_xlim([y_test.min(), y_test.max()])
    ax.set_ylim([y_test.min(), y_test.max()])


def create_parity_plot(model, scaler, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure the model is on the correct device
    model.eval()

    outputs, test_targets = [], []
    with torch.no_grad():
        for i, (onehot_input, embedding_input, target_values) in enumerate(test_loader):
            # Move input tensors to the same device as the model
            onehot_input = onehot_input.to(device)
            target_values = target_values.to(device)

            if type(embedding_input) == list:
                embedding_input = torch.stack(embedding_input, dim=0).squeeze().unsqueeze(0)
            embedding_input = embedding_input.to(device)

            output = model(onehot_input, embedding_input)
            if output.shape[1] == 1:
                rescaled_output = scaler.inverse_transform(output.cpu().reshape(-1, 1))  # Move output back to CPU for scaling
                test_targets.append(scaler.inverse_transform(target_values.cpu().reshape(-1, 1)).reshape(-1).tolist())  # Move target back to CPU for scaling
            else:
                rescaled_output = scaler.inverse_transform(output.cpu())
                test_targets.append(scaler.inverse_transform(target_values.cpu()).reshape(-1).tolist())  # Move target back to CPU for scaling
            outputs.append(rescaled_output.reshape(-1).tolist())

    num_targets = output.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=num_targets, figsize=(4*num_targets, 4))
    for i in range(num_targets):
        ax = axes[i] if num_targets > 1 else axes
        parity_plot(np.array(test_targets)[:, i], np.array(outputs)[:, i], ax, scale='linear')
    fig.tight_layout()


def save_parity_plot(model, scaler, test_loader, filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)  # Ensure the model is on the correct device
    model.eval()

    outputs, test_targets = [], []
    with torch.no_grad():
        for i, (onehot_input, embedding_input, target_values) in enumerate(test_loader):
            # Move input tensors to the same device as the model
            onehot_input = onehot_input.to(device)
            target_values = target_values.to(device)

            if type(embedding_input) == list:
                embedding_input = torch.stack(embedding_input, dim=0).squeeze().unsqueeze(0)
            embedding_input = embedding_input.to(device)

            output = model(onehot_input, embedding_input)
            if output.shape[0] == 1:
                rescaled_output = scaler.inverse_transform(output.cpu().reshape(-1, 1))  # Move output back to CPU for scaling
                test_targets.append(scaler.inverse_transform(target_values.cpu().reshape(-1, 1)).reshape(-1).tolist())  # Move target back to CPU for scaling
            else:
                rescaled_output = scaler.inverse_transform(output.cpu())
                test_targets.append(scaler.inverse_transform(target_values.cpu()).reshape(-1).tolist())  # Move target back to CPU for scaling
            outputs.append(rescaled_output.reshape(-1).tolist())

    num_targets = output.shape[1]
    fig, axes = plt.subplots(nrows=1, ncols=num_targets, figsize=(4*num_targets, 4))
    for i in range(num_targets):
        ax = axes[i] if num_targets > 1 else axes
        parity_plot(np.array(test_targets)[:, i], np.array(outputs)[:, i], ax, scale='linear')

    fig.savefig(filename, dpi=300, bbox_inches='tight')
    model.train()


def prediction_loss(train_losses: list, epoch: int, filename: str = './training_loss.png', test_losses: list = None, injection_interval: int = None):
    """
    Plot and save the training and test losses over epochs.

    Args:
        train_losses (list): List of training losses.
        epoch (int): Number of epochs.
        filename (str): Filepath to save the plot. Default is './training_loss.png'.
        test_losses (list, optional): List of test losses. Default is None.
        injection_interval (int, optional): The interval over which noise is injected for training. 
    Returns:
        None
    """
    fig, ax = plt.subplots(figsize=(5, 3.5), dpi=100)
    ax.plot(list(range(epoch+1)), train_losses, color='blue', label='train loss')
    if test_losses is not None:
        ax.plot(list(range(injection_interval, epoch+1)), test_losses, color='red', label='test loss')
    ax.set(xlabel='epoch', ylabel='Loss')
    ax.legend()
    fig.savefig(filename, facecolor='w')
    plt.close()