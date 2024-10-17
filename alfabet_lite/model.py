import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import rdkit.Chem
from tqdm import tqdm
from pooch import retrieve, Untar

import tensorflow as tf

from . import _model_files_baseurl
from .fragment import Molecule, get_fragments
from .preprocessor import get_features, preprocessor

"""
Merged some functions from predict.py into this file and dropped prediction.py.
"""

model_files = retrieve(
    _model_files_baseurl + "model.tar.gz",
    known_hash="sha256:f1c2b9436f2d18c76b45d95140e6"
    "a08c096250bd5f3e2b412492ca27ab38ad0c",
    processor=Untar(extract_dir="model"),
)

model = tf.keras.models.load_model(os.path.dirname(model_files[0]))

bde_dft = pd.read_csv(
    retrieve(
        _model_files_baseurl + "bonds_for_neighbors.csv.gz",
        known_hash="sha256:d4fb825c42d790d4b2b4bd5dc2d"
        "87c844932e2da82992a31d7521ce51395adb1",
    )
)

def validate_inputs(inputs: Dict) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Check the given SMILES to ensure it's present in the model's
    preprocessor dictionary.

    Returns:
    (is_outlier, missing_atom, missing_bond)

    """
    inputs = {key: np.asarray(val) for key, val in inputs.items()}

    missing_bond = np.array(list(set(inputs["bond_indices"][inputs["bond"] == 1])))
    missing_atom = np.arange(len(inputs["atom"]))[inputs["atom"] == 1]

    is_outlier = (missing_bond.size != 0) | (missing_atom.size != 0)

    return is_outlier, missing_atom, missing_bond



def get_max_bonds(molecule_list: List[Molecule]):
    def num_bonds(molecule):
        molH = rdkit.Chem.AddHs(molecule.mol)
        return molH.GetNumBonds()

    return max((num_bonds(molecule) for molecule in molecule_list))


def predict(smiles_list, drop_duplicates=True, batch_size=1, verbose=False):
    """Predict the BDEs of each bond in a list of molecules.

    Parameters
    ----------
    smiles_list : list
        List of SMILES strings for each molecule
    drop_duplicates : bool, optional
        Whether to drop duplicate bonds (those with the same resulting radicals)
    verbose : bool, optional
        Whether to show a progress bar

    Returns
    -------
    pd.DataFrame
    dataframe of prediction results with columns:

        molecule - SMILES of parent
        bond_index - integer corresponding to given bond (of mol with explicit
                     H's)
        bond_type - elements of start and end atom types
        fragment1 - SMILES of one radical product
        fragment2 - SMILES of second radical product
        delta_assigned_stereo - # of assigned stereocenters created or destroyed
        delta_unassigned_stereo - # of unassigned stereocenters changed
        bde_pred - predicted BDE (in kcal/mol)
        is_valid - whether the starting molecule is present in the model's
                   domain of validity
    """

    molecule_list = [Molecule(smiles=smiles) for smiles in smiles_list]
    smiles_list = [mol.smiles for mol in molecule_list]

    pred_df = pd.concat(
        (
            get_fragments(mol, drop_duplicates=drop_duplicates)
            for mol in tqdm(molecule_list, disable=not verbose)
        )
    )

    max_bonds = get_max_bonds(molecule_list)
    input_dataset = tf.data.Dataset.from_generator(
        lambda: (
            get_features(mol.smiles, max_num_edges=2 * max_bonds)
            for mol in tqdm(molecule_list, disable=not verbose)
        ),
        output_signature=preprocessor.output_signature,
    ).cache()

    batched_dataset = input_dataset.padded_batch(batch_size=batch_size).prefetch(
        tf.data.experimental.AUTOTUNE
    )

    bdes, bdfes = model.predict(batched_dataset, verbose=1 if verbose else 0)

    bde_df = (
        pd.DataFrame(bdes.squeeze(axis=-1), index=smiles_list)
        .T.unstack()
        .reindex(pred_df[["molecule", "bond_index"]])
    )
    bdfe_df = (
        pd.DataFrame(bdfes.squeeze(axis=-1), index=smiles_list)
        .T.unstack()
        .reindex(pred_df[["molecule", "bond_index"]])
    )

    pred_df["bde_pred"] = bde_df.values
    pred_df["bdfe_pred"] = bdfe_df.values

    is_valid = pd.Series(
        {
            smiles: not validate_inputs(input_)[0]
            for smiles, input_ in zip(smiles_list, input_dataset)
        },
        name="is_valid",
    )

    pred_df = pred_df.merge(is_valid, left_on="molecule", right_index=True, how="left")
    pred_df = pred_df.merge(
        bde_dft[["molecule", "bond_index", "bde", "bdfe", "set"]],
        on=["molecule", "bond_index"],
        how="left",
    )

    return pred_df

