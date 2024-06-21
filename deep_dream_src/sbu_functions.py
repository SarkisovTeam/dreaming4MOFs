import os, sys
import pandas as pd
import selfies as sf
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from typing import Optional, List
import pormake as pm 
## TODO: REMOVE DEPENDENCY ON TORCH HERE 
# import torch
import copy
import ase.visualize
try:
    from ase.utils import natural_cutoffs
except Exception as e:
    from ase.neighborlist import natural_cutoffs
from pathlib import Path

#************************ UTILITY FUNCTIONS **********************************#
class BuildingBlock:
    def __init__(self, bb_file, local_structure_func=None):
        self.atoms = read_budiling_block_xyz(bb_file)
        self.name = self.atoms.info["name"]
        self.connection_point_indices = np.array(self.atoms.info["cpi"])
        self._bonds = self.atoms.info["bonds"]
        self._bond_types = self.atoms.info["bond_types"]
        self._has_metal = None
        self.local_structure_func = local_structure_func
        self.check_bonds()

    def copy(self):
        return copy.deepcopy(self)

    def make_chiral_building_block(self):
        _bb = self.copy()
        _bb.atoms.set_positions(-_bb.atoms.get_positions())
        return _bb

    def set_centroid(self, centroid):
        """
        Set centroid of connection points.
        """
        positions = self.atoms.positions
        # Move centroid to zero.
        positions = positions - self.centroid
        # Recentroid by given value.
        positions = positions + centroid
        self.atoms.set_positions(positions)

    @property
    def centroid(self):
        centroid = np.mean(self.connection_points, axis=0)
        return centroid

    @property
    def connection_points(self):
        return self.atoms[self.connection_point_indices].positions

    @property
    def n_connection_points(self):
        return len(self.connection_point_indices)

    @property
    def lengths(self):
        dists = self.connection_points - self.centroid
        norms = np.linalg.norm(dists, axis=1)
        return norms

    @property
    def has_metal(self):
        if self._has_metal is None:
            inter = set(self.atoms.symbols) & set(METAL_LIKE)
            return len(inter) != 0
        else:
            print("has_metal returns self._has_metal.")
            return self._has_metal

    @has_metal.setter
    def has_metal(self, v):
        if isinstance(v, bool) or (v is None):
            print("New value is assigned for self._has_metal.")
            self._has_metal = v
        else:
            raise Exception("Invalid value for has_metal.")

    @property
    def is_edge(self):
        return self.n_connection_points == 2

    @property
    def is_node(self):
        return not self.is_edge

    @property
    def bonds(self):
        if self._bonds is None:
            self.calculate_bonds()

        return self._bonds

    @property
    def bond_types(self):
        if self._bond_types is None:
            self.calculate_bonds()

        return self._bond_types

    @property
    def n_atoms(self):
        return len(self.atoms)

    def check_bonds(self):
        if self._bonds is None:
            self.calculate_bonds()

        # Check whether all atoms has bond or not.
        indices = set(np.array(self._bonds).reshape(-1))
        #X_indices = set([a.index for a in self.atoms if a.symbol == "X"])
        atom_indices = set([a.index for a in self.atoms])

        sub = list(atom_indices - indices)

        if len(sub) != 0:
            pair = [(i, self.atoms.symbols[i]) for i in sub]
            print(
                "There are atoms without bond: %s, %s.", self.name, pair,
            )
            #logger.warning("Make new bond for X.")

    def calculate_bonds(self):
        print("Start calculating bonds.")

        r = self.atoms.positions
        c = 1.2*np.array(natural_cutoffs(self.atoms))

        diff = r[np.newaxis, :, :] - r[:, np.newaxis, :]
        norms = np.linalg.norm(diff, axis=-1)
        cutoffs = c[np.newaxis, :] + c[:, np.newaxis]

        IJ = np.argwhere(norms < cutoffs)
        I = IJ[:, 0]
        J = IJ[:, 1]

        indices = I < J

        I = I[indices]
        J = J[indices]

        self._bonds = np.stack([I, J], axis=1)
        self._bond_types = ["S" for _ in self.bonds]

    def view(self, *args, **kwargs):
        ase.visualize.view(self.atoms, *args, **kwargs)

    def __repr__(self):
        msg = "BuildingBlock: {}, # of connection points: {}".format(
            self.name, self.n_connection_points
        )
        return msg

    def write_cif(self, filename):
        write_molecule_cif(filename, self.atoms, self.bonds, self.bond_types)


def write_molecule_cif(filename, atoms, bond_pairs, bond_types):
    """
    Write cif for the molecule structures.

    Args:
        filename: file name.
        atoms: ase.Atoms object.
        bond_pairs: list of bond paris. contains (i, j).
        bond_types: list of bond types. contains one of "S", "D", "T", "A".

    Returns:
        None
    """

    path = Path(filename).resolve()
    if path.suffix != ".cif":
        path = path.with_suffix(".cif")

    stem = path.stem.replace(" ", "_")
    with path.open("w") as f:
        f.write("data_{}\n".format(stem))

        f.write("_symmetry_space_group_name_H-M    P1\n")
        f.write("_symmetry_Int_Tables_number       1\n")
        f.write("_symmetry_cell_setting            triclinic\n")

        f.write("loop_\n")
        f.write("_symmetry_equiv_pos_as_xyz\n")
        f.write("'x, y, z'\n")

        # Calculate cell parameters.
        positions = atoms.get_positions()
        com = atoms.get_center_of_mass()

        distances = np.linalg.norm(positions - com, axis=1)
        max_distances = np.max(distances)

        box_length = 2*max_distances + 4

        f.write("_cell_length_a     {:.3f}\n".format(box_length))
        f.write("_cell_length_b     {:.3f}\n".format(box_length))
        f.write("_cell_length_c     {:.3f}\n".format(box_length))
        f.write("_cell_angle_alpha  90.0\n")
        f.write("_cell_angle_beta   90.0\n")
        f.write("_cell_angle_gamma  90.0\n")

        f.write("loop_\n")
        f.write("_atom_site_label\n")
        f.write("_atom_site_type_symbol\n")
        f.write("_atom_site_fract_x\n")
        f.write("_atom_site_fract_y\n")
        f.write("_atom_site_fract_z\n")
        f.write("_atom_type_partial_charge\n")

        # Get fractional coordinates
        # fractional coordinate of C.O.M is (0.5, 0.5, 0.5).
        symbols = atoms.get_chemical_symbols()
        fracts = (positions - com) / box_length + 0.5

        # Write label and pos information.
        for i, (sym, fract) in enumerate(zip(symbols, fracts)):
            label = "{}{}".format(sym, i)
            f.write("{} {} {:.5f} {:.5f} {:.5f} 0.0\n".
                    format(label, sym, *fract))

        # Write bonds information.
        f.write("loop_\n")
        f.write("_geom_bond_atom_site_label_1\n")
        f.write("_geom_bond_atom_site_label_2\n")
        f.write("_geom_bond_distance\n")
        f.write("_geom_bond_site_symmetry_2\n")
        f.write("_ccdc_geom_bond_type\n")

        for (i, j), t in zip(bond_pairs, bond_types):
            label_i = "{}{}".format(symbols[i], i)
            label_j = "{}{}".format(symbols[j], j)

            distance = np.linalg.norm(positions[i]-positions[j])

            f.write("{} {} {:.3f} . {}\n".
                format(label_i, label_j, distance, t)
            )

# Metal species.
METAL_LIKE = [
    "Li", "Be", "B", "Na", "Mg",
    "Al", "Si", "K", "Ca", "Sc",
    "Ti", "V", "Cr", "Mn", "Fe",
    "Co", "Ni", "Cu", "Zn", "Ga",
    "Ge", "As", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In",
    "Sn", "Sb", "Te", "Cs", "Ba",
    "La", "Ce", "Pr", "Nd", "Pm",
    "Sm", "Eu", "Gd", "Tb", "Dy",
    "Ho", "Er", "Tm", "Yb", "Lu",
    "Hf", "Ta", "W", "Re", "Os",
    "Ir", "Pt", "Au", "Hg", "Tl",
    "Pb", "Bi", "Po", "Fr", "Ra",
    "Ac", "Th", "Pa", "U", "Np",
    "Pu", "Am", "Cm", "Bk", "Cf",
    "Es", "Fm", "Md", "No", "Lr",
]


def read_budiling_block_xyz(bb_file):
    '''
    Return an ase atoms object from an xyz file

    '''
    name = Path(bb_file).stem

    with open(bb_file, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0])

    symbols = []
    positions = []
    connection_point_indices = []
    for i, line in enumerate(lines[2 : n_atoms+2]):
        tokens = line.split()
        symbol = tokens[0]
        position = [float(v) for v in tokens[1:]]

        symbols.append(symbol)
        positions.append(position)
        if symbol == "X":
            connection_point_indices.append(i)

    bonds = None
    bond_types = None
    if len(lines) > n_atoms+2:
        bonds = []
        bond_types = []

        for line in lines[n_atoms+2:]:
            tokens = line.split()
            if len(tokens) < 3:
                continue

            i = int(tokens[0])
            j = int(tokens[1])
            t = tokens[2]
            bonds.append((i, j))
            bond_types.append(t)
        bonds = np.array(bonds)

    info = {}
    info["cpi"] = connection_point_indices
    info["name"] = name
    info["bonds"] = bonds
    info["bond_types"] = bond_types

    atoms = ase.Atoms(symbols=symbols, positions=positions, info=info)

    return atoms


def get_smiles_for_bb(bb):
    """
    It takes a building block object from PORMAKE and returns a SMILES string
    
    :param bb: The building block object
    :return: A string of the SMILES representation of the molecule.
    """

    import pysmiles as ps
    import networkx as nx
    mol = nx.Graph()
    mol.add_edges_from(bb.bonds)
    nx.set_node_attributes(mol, dict(enumerate(bb.atoms.get_chemical_symbols())), 'element')
    nx.set_node_attributes(mol, dict(enumerate(bb.atoms.get_positions())), 'positions')
    # nx.set_node_attributes(mol, best_node.atoms.get_positions(), 'positions')
    ps.fill_valence(mol, respect_hcount=True, respect_bond_order=False)
    return ps.write_smiles(mol)

    

def write_bb_as_xyz(smiles, path_to_database, bb_name, connector_char='[Fr]', closeness=0.75):
    """
    It takes a SMILES string, a path to the database, a name for the building block, and a character
    that represents the connection points in the SMILES string, and writes an xyz file for the building
    block in the database
    
    :param smiles: the SMILES string of the building block
    :param path_to_database: the path to the database you want to write the building block to
    :param bb_name: the name of the building block. This will be the name of the file that is created
    :param connector_char: the character that is used to represent the connection points in the SMILES
    string, defaults to [Fr] (optional)
    :param closeness: how close to the base atom should the connection point be?
    """

    #* To get the length of the connection points right
    # mol1 = Chem.MolFromSmiles(smiles.replace(connector_char, '[Li]'))
    mol1 = Chem.MolFromSmiles(smiles)
    mol1 = Chem.AddHs(mol1)
    AllChem.EmbedMolecule(mol1)
    out1= Chem.MolToXYZBlock(mol1).split('\n')
    # print('\n'.join(out1))

    # edit = Chem.RWMol(mol)
    conf = mol1.GetConformer()
    # print(Chem.MolToXYZBlock(mol=mol1))

    # * let's bring the connection points closer to the base atoms
    # * so that bonds will be formed for assembling the mofs

    rep_ids = []
    fr_ids = []
    for atom in mol1.GetAtoms():
        if atom.GetAtomicNum()==87:
            fr_ids.append(atom.GetIdx())
            for n in atom.GetNeighbors():
                rep_ids.append(n.GetIdx())

    positions = conf.GetPositions()
    fr_positions = positions[fr_ids]
    base_positions = positions[rep_ids]

    new_positions = fr_positions + closeness*(base_positions - fr_positions)
    from rdkit.Geometry import Point3D
    conf = mol1.GetConformer()
    for i in range(len(new_positions)):
        x,y,z = new_positions[i]
        conf.SetAtomPosition(fr_ids[i],Point3D(x,y,z))
    mol1.AddConformer(conf)
    out1 = Chem.MolToXYZBlock(mol=mol1, confId=-1).split('\n')

    
    # # * To find the index of the connection points if you embed with Fr the coordinates would be wrong!
    # mol_find_X = Chem.MolFromSmiles(smiles.replace(connector_char, '[Fr]'))
    # AllChem.EmbedMolecule(mol_find_X)
    # out_find_X= Chem.MolToXYZBlock(mol_find_X).split('\n')
    connection_point_lines = [str(k) for k in np.where([connector_char.replace('[','').replace(']','') in o for o in out1])[0]-2]
    # connection_point_lines = [str(k) for k in np.where( ['Li' in o for o in out1])[0]-2]
    # print(connection_point_lines)
    # * Now take the coordinates from out1 and replace H in 
    # * connection point lines with X
    out1[1] = '\t' + '\t'.join(connection_point_lines)
    # for xline in connection_point_lines:
    # print(fr_ids)
    # print(rep_ids)
    # print(positions)
    # print( fr_positions)
    # print(base_positions)
    # print( new_positions)
    # print( conf.GetPositions())
    #     out1[int(xline)] = out1[int(xline)].replace('Li','X')
    
    # print('\n'.join(out1).replace('Li','X'))
    # print('\n'.join(out1).replace(connector_char.replace('[','').replace(']',''),'X'))
    # print('\n'.join(out_find_X))
    filename = os.path.join(os.path.abspath(path_to_database),'bbs',bb_name + '.xyz')
    print(filename)
    
    bbs_path = os.path.join(path_to_database, 'bbs')
    if not os.path.exists(bbs_path):
        os.makedirs(bbs_path)
    with open(filename, 'w') as f:
        f.write('\n'.join(out1).replace(connector_char.replace('[','').replace(']',''),'X'))
        # f.write('\n'.join(out1).replace('Li','X'))
        # f.write('\n')
        for bond in mol1.GetBonds():
            f.write('\t'+str(bond.GetBeginAtomIdx())+'\t'+ str(bond.GetEndAtomIdx()) +' '+ str(bond.GetBondType())[0]+'\n')

def write_bb_as_xyz2(smiles, path_to_database, bb_name, connector_char='[Fr]', closeness=0.75):
    """
    It takes a SMILES string, a path to the database, a name for the building block, and a character
    that represents the connection points in the SMILES string, and writes an xyz file for the building
    block in the database
    
    :param smiles: the SMILES string of the building block
    :param path_to_database: the path to the database you want to write the building block to
    :param bb_name: the name of the building block. This will be the name of the file that is created
    :param connector_char: the character that is used to represent the connection points in the SMILES
    string, defaults to [Fr] (optional)
    :param closeness: how close to the base atom should the connection point be?
    """

    from rdkit import Chem
    from rdkit.Chem import AllChem
    import os
    filename = os.path.join(os.path.abspath(path_to_database) , 'bbs', bb_name + '.xyz')
    #* To get the length of the connection points right
    # mol1 = Chem.MolFromSmiles(smiles.replace(connector_char, '[Li]'))
    mol1 = Chem.MolFromSmiles(smiles)
    mol1 = Chem.AddHs(mol1)
    # AllChem.EmbedMolecule(mol1)
    AllChem.EmbedMolecule(mol1, AllChem.ETKDG())
    AllChem.UFFOptimizeMolecule(mol1)
    out1= Chem.MolToXYZBlock(mol1).split('\n')
    # print('\n'.join(out1))

    # edit = Chem.RWMol(mol)
    conf = mol1.GetConformer()
    # print(Chem.MolToXYZBlock(mol=mol1))

    # * let's bring the connection points closer to the base atoms
    # * so that bonds will be formed for assembling the mofs

    rep_ids = []
    fr_ids = []
    for atom in mol1.GetAtoms():
        if atom.GetAtomicNum()==87:
            fr_ids.append(atom.GetIdx())
            for n in atom.GetNeighbors():
                rep_ids.append(n.GetIdx())

    positions = conf.GetPositions()
    fr_positions = positions[fr_ids]
    base_positions = positions[rep_ids]

    new_positions = fr_positions + closeness*(base_positions - fr_positions)
    from rdkit.Geometry import Point3D
    conf = mol1.GetConformer()
    for i in range(len(new_positions)):
        x,y,z = new_positions[i]
        conf.SetAtomPosition(fr_ids[i],Point3D(x,y,z))
    mol1.AddConformer(conf)
    out1 = Chem.MolToXYZBlock(mol=mol1, confId=-1).split('\n')

    
    # # * To find the index of the connection points if you embed with Fr the coordinates would be wrong!
    # mol_find_X = Chem.MolFromSmiles(smiles.replace(connector_char, '[Fr]'))
    # AllChem.EmbedMolecule(mol_find_X)
    # out_find_X= Chem.MolToXYZBlock(mol_find_X).split('\n')
    connection_point_lines = [str(k) for k in np.where([connector_char.replace('[','').replace(']','') in o for o in out1])[0]-2]
    # connection_point_lines = [str(k) for k in np.where( ['Li' in o for o in out1])[0]-2]
    # print(connection_point_lines)
    # * Now take the coordinates from out1 and replace H in 
    # * connection point lines with X
    out1[1] = '\t' + '\t'.join(connection_point_lines)
    # for xline in connection_point_lines:
    # print(fr_ids)
    # print(rep_ids)
    # print(positions)
    # print( fr_positions)
    # print(base_positions)
    # print( new_positions)
    # print( conf.GetPositions())
    #     out1[int(xline)] = out1[int(xline)].replace('Li','X')
    
    # print('\n'.join(out1).replace('Li','X'))
    print('\n'.join(out1).replace(connector_char.replace('[','').replace(']',''),'X'))
    # print('\n'.join(out_find_X))
    bbs_path = os.path.join(path_to_database, 'bbs')
    if not os.path.exists(bbs_path):
        os.makedirs(bbs_path)
    with open(filename, 'w') as f:
        f.write('\n'.join(out1).replace(connector_char.replace('[','').replace(']',''),'X'))
        # f.write('\n'.join(out1).replace('Li','X'))
        # f.write('\n')
        for bond in mol1.GetBonds():
            f.write('\t'+str(bond.GetBeginAtomIdx())+'\t'+ str(bond.GetEndAtomIdx()) +' '+ str(bond.GetBondType())[0]+'\n')

def write_smiles_as_bb(smiles, path_to_database, bb_name, closeness=0.75):
    import os
    import numpy as np
    from rdkit import Chem
    from sklearn.decomposition import PCA
    from rdkit.Chem import rdMolTransforms

    # Parse the SMILES string to generate a molecule
    mol = Chem.MolFromSmiles(smiles)

    # Add hydrogens to the molecule
    mol = Chem.AddHs(mol)

    # Generate 3D coordinates
    Chem.AllChem.EmbedMolecule(mol)

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

    # Remove hydrogens
    # mol = Chem.RemoveHs(mol)

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
    ''''
    Construct MOF 

    Inputs: 
        topo (str): pormake topology code
        node (str): pormake topology code
        edge_smi (Optional[str]): smiles string representing a MOF linker
        edge_smi_name (Optional[str]): name of MOF linker 
        path_to_edge_smi (Optional[str]): file path to xyz file which represents a MOF linker
        save_mof_to_dir (Optional, str): name of directory to save MOF cif file to
        cif_file_name (str): name of cif file 
    '''
    if not os.path.exists(save_mof_to_dir):
        os.makedirs(save_mof_to_dir)
    
    db = pm.Database()
    builder = pm.Builder()

    if edge_smi is None:
        if path_to_edge_xyz is None:
            raise ValueError(
                "Either a string for edge_smi or a filepath for path_to_edge_xyz must be provided"
                )
        else:
            edge_bb = pm.BuildingBlock(path_to_edge_xyz)
    else:
        # canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(edge_smi))
        #TODO NEED TO SORT OUT WHATEVER ISSUE IS CAUSING AN ERROR HERE
        canon_smi = Chem.MolToSmiles(Chem.MolFromSmiles(edge_smi))
        # write_bb_as_xyz(
        #     smiles = canon_smi,
        #     path_to_database = save_mof_to_dir,
        #     bb_name = edge_smi_name if not None else canon_smi,
        #     )
        write_smiles_as_bb(
            smiles = canon_smi,
            path_to_database = save_mof_to_dir,
            bb_name = edge_smi_name if not None else canon_smi,
            )
        edge_bb = pm.BuildingBlock(f"{save_mof_to_dir}/bbs/{edge_smi_name if not None else canon_smi}.xyz")

    node_bb = db.get_bb(node)
    topo = db.get_topo(topology)

    topo_0 = topo.unique_local_structures[0]
    locator = pm.Locator()
    topo_rmsd_0 = locator.calculate_rmsd(topo_0, node_bb)
    if topo_rmsd_0 < 0.3:
        print("RMSD at random node type 0: %.2f" % topo_rmsd_0)
        mof = builder.build_by_type(topology=topo, node_bbs={0: node_bb}, edge_bbs={(0, 0): edge_bb})

        if save_mof_to_dir is not None:
            print(cif_file_name)
            # mof.write_cif(os.path.abspath())
            mof.write_cif(os.path.join(save_mof_to_dir,cif_file_name))
    else:
        print('Failed')
