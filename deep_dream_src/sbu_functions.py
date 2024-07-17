import os
import numpy as np
from rdkit import Chem
from typing import Optional
import pormake as pm 
import pysmiles as ps
import networkx as nx
from sklearn.decomposition import PCA
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import AllChem
import uuid

def generate_truncated_uuids(n, length=8):
    """
    Generate a list of truncated UUIDs to uniquely identify generated linkers.

    Parameters:
    - n (int): The number of UUIDs to generate.
    - length (int): The length of the truncated UUIDs. Default is 8.

    Returns:
    - list: A list of truncated UUIDs.
    """
    ids = []
    for _ in range(n):
        unique_id = str(uuid.uuid4()).replace('-', '')[:length]
        ids.append(unique_id)
    return ids


def get_smiles_for_bb(bb):
    """
    It takes a building block object from PORMAKE and returns a SMILES string
    
    Parameters:
    - bb (Object): The building block object
    
    Returns: 
    - bb_smiles (str): A string of the SMILES representation of the molecule.
    """
    mol = nx.Graph()
    mol.add_edges_from(bb.bonds)
    nx.set_node_attributes(mol, dict(enumerate(bb.atoms.get_chemical_symbols())), 'element')
    nx.set_node_attributes(mol, dict(enumerate(bb.atoms.get_positions())), 'positions')
    # nx.set_node_attributes(mol, best_node.atoms.get_positions(), 'positions')
    ps.fill_valence(mol, respect_hcount=True, respect_bond_order=False)
    bb_smiles = ps.write_smiles(mol)
    return bb_smiles

    
def write_smiles_as_bb(smiles, path_to_database, bb_name, closeness=0.75):
    """
    Writes the given SMILES string as a building block (BB) in an XYZ file format.

    Parameters:
    - smiles (str): The SMILES string representing the molecule.
    - path_to_database (str): The path to the database where the BB will be saved.
    - bb_name (str): The name of the BB.
    - closeness (float, optional): The desired bond length of Fr-bonded neighbors. Default is 0.75.

    Returns:
    - None
    """
    # Parse the SMILES string to generate a molecule
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    AllChem.EmbedMolecule(mol)

    # Identify the rings in the molecule
    ssr = Chem.GetSymmSSSR(mol)

    # For each ring
    for ring in ssr:
        # Get the coordinates of the atoms in the ring
        coords = np.array([mol.GetConformer().GetAtomPosition(i) for i in ring])

        # Calculate the average plane using PCA
        pca = PCA(n_components=3)
        pca.fit(coords)

        # The last component of PCA is the normal to the plane
        normal = pca.components_[-1]

        # Calculate the distance from each atom to the plane
        distances = np.dot(coords - pca.mean_, normal)

        # Shift the atoms to the plane
        for i, atom_idx in enumerate(ring):
            pos = mol.GetConformer().GetAtomPosition(atom_idx)
            new_pos = [pos.x - distances[i] * normal[0], pos.y - distances[i] * normal[1], pos.z - distances[i] * normal[2]]
            mol.GetConformer().SetAtomPosition(atom_idx, new_pos)

        # Calculate the centroid of the ring
        centroid = np.mean(coords, axis=0)

        # Adjust the position of the atoms single bonded to the ring
        for atom in mol.GetAtoms():
            if atom.GetDegree() == 1:  # Single bonded atoms
                for neighbor in atom.GetNeighbors():
                    if neighbor.GetIdx() in ring:
                        # Get the position of the neighbor atom
                        neighbor_pos = mol.GetConformer().GetAtomPosition(neighbor.GetIdx())

                        # Calculate the direction vector from the centroid to the neighbor atom
                        direction = np.array([neighbor_pos.x, neighbor_pos.y, neighbor_pos.z]) - centroid

                        # Normalize the direction vector
                        direction /= np.linalg.norm(direction)

                        # Calculate the new position of the atom
                        new_pos = np.array([neighbor_pos.x, neighbor_pos.y, neighbor_pos.z]) + 1.5 * direction

                        # Set the new position of the atom
                        mol.GetConformer().SetAtomPosition(atom.GetIdx(), new_pos)

    # Adjust the bond length of Fr-bonded neighbors to 0.75 A
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == 'Fr':
            for neighbor in atom.GetNeighbors():
                rdMolTransforms.SetBondLength(mol.GetConformer(), atom.GetIdx(), neighbor.GetIdx(), closeness)

    # Create 'bbs' directory if it does not exist
    bbs_dir = os.path.join(path_to_database, 'bbs')
    if not os.path.exists(bbs_dir):
        os.makedirs(bbs_dir)

    # Write to an XYZ file
    xyz_file = os.path.join(bbs_dir, f'{bb_name}.xyz')
    print(xyz_file)
    with open(xyz_file, 'w') as f:
        # Write the number of atoms
        f.write(f"{mol.GetNumAtoms()}\n")

        # Write a blank line or a comment
        f.write('\t'+'\t'.join([str(i.GetIdx()) for i in mol.GetAtoms() if i.GetSymbol() == 'Fr']) + "\n")

        # Write the atoms
        for atom in mol.GetAtoms():
            pos = mol.GetConformer().GetAtomPosition(atom.GetIdx())
            # If the atom symbol is 'Fr', write 'X' instead
            symbol = 'X' if atom.GetSymbol() == 'Fr' else atom.GetSymbol()
            f.write(f"{symbol} \t{np.round(pos.x,decimals=4)} {np.round(pos.y,decimals=4)} {np.round(pos.z,decimals=4)}\n")

        # Write the bonds
        for bond in mol.GetBonds():
            bond_type = 'S' if bond.GetBondType() == Chem.rdchem.BondType.SINGLE else \
                        'D' if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE else \
                        'T' if bond.GetBondType() == Chem.rdchem.BondType.TRIPLE else \
                        'A' if bond.GetBondType() == Chem.rdchem.BondType.AROMATIC else '?'
            f.write(f"\t{bond.GetBeginAtomIdx()} \t{bond.GetEndAtomIdx()} {bond_type}\n")


def construct_mof(
    topology: str,
    node: str, 
    edge_smi: Optional[str] = None, 
    edge_smi_name: Optional[str] = None,
    path_to_edge_xyz: Optional[str] = None, 
    save_mof_to_dir: str = None,
    cif_file_name: str = 'MOF.cif'
    ):
    """
    Constructs a MOF (Metal-Organic Framework) using the provided topology, node, and edge information.

    Parameters:
    - topology (str): pormake topology code.
    - node (str): pormake node code.
    - edge_smi (Optional[str]): The SMILES string representation of the edge. Default is None.
    - edge_smi_name (Optional[str]): The name of MOF linker, used to name the bb XYZ file. Default is None.
    - path_to_edge_xyz (Optional[str]): The file path to the edge XYZ file. Default is None.
    - save_mof_to_dir (str): The directory to save the MOF. Default is None.
    - cif_file_name (str): The name of the CIF file. Default is 'MOF.cif'.

    Returns:
    - None

    Raises:
    - ValueError: If neither edge_smi nor path_to_edge_xyz is provided.

    """
    # create save directory
    if not os.path.exists(save_mof_to_dir):
        os.makedirs(save_mof_to_dir)
    
    # create database and builder objects
    db = pm.Database()
    builder = pm.Builder()

    # get edge building block
    if edge_smi is None:
        if path_to_edge_xyz is None:
            raise ValueError(
                "Either a string for edge_smi or a filepath for path_to_edge_xyz must be provided"
                )
        else:
            # Either get the building block object directly from the pre-generated XYZ file
            edge_bb = pm.BuildingBlock(path_to_edge_xyz)
    else:
        # or generate the building block object from the SMILES string
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(edge_smi))
        write_smiles_as_bb(
            smiles = canon_smi,
            path_to_database = save_mof_to_dir,
            bb_name = edge_smi_name if not None else canon_smi,
            )
        edge_bb = pm.BuildingBlock(f"{save_mof_to_dir}/bbs/{edge_smi_name if not None else canon_smi}.xyz")

    # get node building block and topology
    node_bb = db.get_bb(node)
    topo = db.get_topo(topology)

    # construct MOF
    topo_0 = topo.unique_local_structures[0]
    locator = pm.Locator()
    topo_rmsd_0 = locator.calculate_rmsd(topo_0, node_bb)
    if topo_rmsd_0 < 0.3:
        print("RMSD at random node type 0: %.2f" % topo_rmsd_0)
        mof = builder.build_by_type(topology=topo, node_bbs={0: node_bb}, edge_bbs={(0, 0): edge_bb})

        if save_mof_to_dir is not None:
            print(cif_file_name)
            mof.write_cif(os.path.join(save_mof_to_dir,cif_file_name))
    else:
        print('Failed')
