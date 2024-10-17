from rdkit import Chem
from rdkit.Chem import rdDepictor
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Geometry.rdGeometry import Point2D
import os
import numpy as np

def draw_bde(smiles, bond_index, predicted_bde=None, actual_bde=None,figwidth=200):
    """Draw a molecule with a highlighted bond.
    Modified in Lite to also visualize its predicted BDE.

    Parameters:
    smiles (str): SMILES representation of the molecule.
    bond_index (int): Index of the bond to highlight.
    predicted_bde (float, optional): Predicted bond dissociation energy.
    actual_bde (float, optional): Actual bond dissociation energy.
    figwidth (int, optional): Width of the figure.
    
    """    
    mol = Chem.MolFromSmiles(smiles)
    bond_index = int(bond_index)

    if mol.GetNumAtoms() > 20:
        figwidth = 300
    if mol.GetNumAtoms() > 40:
        figwidth = 400

    if bond_index >= mol.GetNumBonds():
        molH = Chem.AddHs(mol)
        if bond_index >= molH.GetNumBonds():
            raise RuntimeError(
                f"Fewer than {bond_index} bonds in {smiles}: "
                f"{molH.GetNumBonds()} total bonds"
            )
        bond = molH.GetBondWithIdx(bond_index)

        start_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
        mol = Chem.AddHs(mol, onlyOnAtoms=[start_atom.GetIdx()])
        bond_index = mol.GetNumBonds() - 1

    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(figwidth, figwidth)
    drawer.drawOptions().fixedBondLength = 30
    drawer.drawOptions().highlightBondWidthMultiplier = 20

    drawer.DrawMolecule(
        mol,
        highlightAtoms=[],
        highlightBonds=[
            bond_index,
        ],
    )

    text_pos = Point2D(0,5)
    if predicted_bde is not None:        
        prediction_text = f"Predicted BDE: {predicted_bde:.2f} kcal/mol"
        if actual_bde is not None and actual_bde != 0 and not np.isnan(actual_bde):
            prediction_text += f"\nActual BDE: {actual_bde:.2f} kcal/mol"
            prediction_text += f"\nError: {abs(predicted_bde - actual_bde):.2f} kcal/mol"

        drawer.DrawString(prediction_text, text_pos, 0)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg

def draw_mol_outlier(smiles, missing_atoms, missing_bonds, figsize=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    missing_bonds_adjusted = []
    for bond_index in missing_bonds:

        if bond_index >= mol.GetNumBonds():
            molH = Chem.AddHs(mol)
            bond = molH.GetBondWithIdx(int(bond_index))

            start_atom = mol.GetAtomWithIdx(bond.GetBeginAtomIdx())
            mol = Chem.AddHs(mol, onlyOnAtoms=[start_atom.GetIdx()])
            bond_index = mol.GetNumBonds() - 1

        missing_bonds_adjusted += [int(bond_index)]

    if not mol.GetNumConformers():
        rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(*figsize)
    drawer.SetFontSize(0.6)
    drawer.DrawMolecule(
        mol,
        highlightAtoms=[int(index) for index in missing_atoms],
        highlightBonds=missing_bonds_adjusted,
    )

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg

def draw_mol(smiles, figsize=(300, 300)):
    mol = Chem.MolFromSmiles(smiles)
    rdDepictor.Compute2DCoords(mol)

    drawer = rdMolDraw2D.MolDraw2DSVG(*figsize)
    drawer.SetFontSize(0.6)
    drawer.DrawMolecule(mol)

    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()

    return svg


def save_svg(svg, save_path):
    """
    To enable svg string saving
    """
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            f.write(svg)
        print(f"SVG saved to {save_path}")