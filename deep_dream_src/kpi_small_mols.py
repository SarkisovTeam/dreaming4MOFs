import os, sys
import pandas as pd
import selfies as sf
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors
## TODO: REMOVE DEPENDENCY ON TORCH HERE 
# import torch
sys.path.append(os.path.join(Chem.RDConfig.RDContribDir, 'SA_Score'))
import sascorer

def logP_from_selfies(list_of_selfies):
    '''
    Calculate list of logP from list of selfies molecules
    '''
    # first convert selfies to smiles
    list_of_smiles = list(map(sf.decoder, list_of_selfies))

    # get logP values
    logP_vals=[]
    for ii in range(len(list_of_smiles)):
        try:
            res_molecule = Chem.MolFromSmiles(list_of_smiles[ii])
        except Exception:
            res_molecule=None

        if res_molecule==None:
            logP_vals.append(-666)
        else:
            logP_vals.append(Descriptors.MolLogP(res_molecule))
    return np.array(logP_vals)


def logP_from_smiles(list_of_smiles):
    '''
    Calculate list of logP from list of selfies molecules
    '''
    logP_vals=[]
    for ii in range(len(list_of_smiles)):
        try:
            res_molecule = Chem.MolFromSmiles(list_of_smiles[ii])
        except Exception:
            res_molecule=None

        if res_molecule==None:
            logP_vals.append(-666)
        else:
            logP_vals.append(Descriptors.MolLogP(res_molecule))
    return np.array(logP_vals)


def SAscore_from_selfies(list_of_selfies):
    '''
    Calculate list of Synthetic accessibility scores from list of selfies molecules
    '''
    # first convert selfies to smiles
    list_of_smiles = list(map(sf.decoder, list_of_selfies))

    # get SA scores 
    SA_vals=[]
    for ii in range(len(list_of_smiles)):
        try:
            res_molecule = Chem.MolFromSmiles(list_of_smiles[ii])
        except Exception:
            res_molecule=None

        if res_molecule==None:
            SA_vals.append(np.nan)
        else:
            SA_vals.append(sascorer.calculateScore(res_molecule))
    return np.array(SA_vals)


def SAscore_from_smiles(list_of_smiles):
    '''
    Calculate list of Synthetic accessibility scores from list of selfies molecules
    '''
    SA_vals =[]
    for ii in range(len(list_of_smiles)):
        try:
            res_molecule = Chem.MolFromSmiles(list_of_smiles[ii])
        except Exception:
            res_molecule=None

        if res_molecule==None:
            SA_vals.append(np.nan)
        else:
            SA_vals.append(sascorer.calculateScore(res_molecule))
    return np.array(SA_vals)