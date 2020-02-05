#!/usr/bin/env python

import numpy as np
import pandas as pd
import mdtraj as mt
import math
import itertools
import sys
from collections import OrderedDict
import argparse
from argparse import RawDescriptionHelpFormatter
import multiprocessing as mp


ALL_ELEMENTS = ["H", "C", "O", "N", "P", "S", "HAX", "DU"]


class AtomTypeCounts(object):
    """Featurization of Protein-Ligand Complex based on
    onion-shape distance counts of atom-types.

    Parameters
    ----------
    pdb_fn : str
        The input pdb file name.
    lig_code : str
        The ligand residue name in the input pdb file.

    Attributes
    ----------
    pdb : mdtraj.Trajectory
        The mdtraj.trajectory object containing the pdb.
    receptor_indices : np.ndarray
        The receptor (protein) atom indices in mdtraj.Trajectory
    ligand_indices : np.ndarray
        The ligand (protein) atom indices in mdtraj.Trajectory
    rec_ele : np.ndarray
        The element types of each of the atoms in the receptor
    lig_ele : np.ndarray
        The element types of each of the atoms in the ligand
    lig_code : str
        The ligand residue name in the input pdb file
    pdb_parsed_ : bool
        Whether the pdb file has been parsed.
    distance_computed : bool
        Whether the distances between atoms in receptor and ligand has been computed.
    distance_matrix_ : np.ndarray, shape = [ N1 * N2, ]
        The distances between all atom pairs
        N1 and N2 are the atom numbers in receptor and ligand respectively.
    counts_: np.ndarray, shape = [ N1 * N2, ]
        The contact numbers between all atom pairs
        N1 and N2 are the atom numbers in receptor and ligand respectively.

    """

    def __init__(self, pdb_fn, lig_code):

        self.pdb = mt.load_pdb(pdb_fn)

        self.receptor_indices = np.array([])
        self.ligand_indices = np.array([])

        self.rec_ele = np.array([])
        self.lig_ele = np.array([])

        self.lig_code = lig_code

        self.pdb_parsed_ = False
        self.distance_computed_ = False

        self.distance_matrix_ = np.array([])
        self.counts_ = np.array([])

    def parsePDB(self, rec_sele="protein", lig_sele="UNK"):
        """
        Parse PDB file using mdtraj

        Parameters
        ----------
        rec_sele: str, default is protein.
            The topology selection for the receptor
        lig_sele: str, default is resname LIG
            The topology selection for the ligand

        Returns
        -------
        self: an instance of itself

        """

        top = self.pdb.topology

        self.receptor_indices = top.select(rec_sele)
        self.ligand_indices = top.select("resname " + lig_sele)

        table, bond = top.to_dataframe()

        self.rec_ele = table['element'][self.receptor_indices]
        self.lig_ele = table['element'][self.ligand_indices]

        self.pdb_parsed_ = True

        return self

    def distance_pairs(self):
        """Calculate all distance pairs between atoms in the receptor and in the ligand

        Returns
        -------
        self: an instance of itself
        """

        if not self.pdb_parsed_:
            self.parsePDB()

        # all combinations of the atom indices from the receptor and the ligand
        all_pairs = itertools.product(self.receptor_indices, self.ligand_indices)

        # if distance matrix is not calculated
        if not self.distance_computed_:
            self.distance_matrix_ = mt.compute_distances(self.pdb, atom_pairs=all_pairs)[0]

        self.distance_computed_ = True

        return self

    def switchFuction(self, x, d0=7.0, m=12, n=6):
        """Implement a rational switch function
        to enable a smooth transition
        Parameters
        ----------
        x : float
            the input distance value
        d0 : float
            the distance cutoff, it is usually 2 times of
            the real distance cutoff
        m : int
            the exponential index of higher order
        n : int
            the exponential index of lower order
        Returns
        -------
        switched_dist : float
            the switched continuous distance
        Notes
        -----
        the function is like:
          s= [1 - (x/d0)^6] / [1 - (x/d0)^12]
        d0 is a cutoff, should be twice larger than the distance cutoff
        """
        count = 0.0
        try:
            count = (1.0 - math.pow((x / d0), n)) / (1.0 - math.pow((x / d0), m))
        except ZeroDivisionError:
            print("Divide by zero, ", x, d0)

        return count

    def cutoff_count(self, cutoff=0.35, switch=False):
        """
        Get the atom contact matrix

        Parameters
        ----------
        cutoff: float, default is 0.35 angstrom
            The distance cuntoff for the contacts

        Returns
        -------
        self: return an instance of itself
        """
        # get the inter-molecular atom contacts
        if switch:
            vect_switch = np.vectorize(self.switchFuction)
            self.distance_matrix_ = vect_switch(self.distance_matrix_)

        self.counts_ = (self.distance_matrix_ <= cutoff) * 1.0

        return self


def get_elementtype(e):
    #all_elements = ["H", "C", "O", "N", "P", "S", "Cl", "DU"]
    if e in ALL_ELEMENTS:
        return e
    elif e in ['Cl', 'Br', 'I', 'F']:
        return 'HAX'
    else:
        return "DU"


def generate_features(complex_fn, lig_code, ncutoffs):

    keys = ["_".join(x) for x in list(itertools.product(ALL_ELEMENTS, ALL_ELEMENTS))]

    # parse the pdb file and get the atom element information
    cplx = AtomTypeCounts(complex_fn, lig_code)
    cplx.parsePDB(rec_sele="protein", lig_sele=lig_code)
    # element types of all atoms in the proteins and ligands
    new_lig = list(map(get_elementtype, cplx.lig_ele))
    new_rec = list(map(get_elementtype, cplx.rec_ele))

    # the element-type combinations for all atom-atom pairs
    rec_lig_element_combines = ["_".join(x) for x in list(itertools.product(new_rec, new_lig))]
    cplx.distance_pairs()

    counts = []
    onion_counts = []

    # calculate all contacts for all shells
    for i, cutoff in enumerate(ncutoffs):
        cplx.cutoff_count(cutoff)
        if i == 0:
            onion_counts.append(cplx.counts_)
        else:
            onion_counts.append(cplx.counts_ - counts[-1])
        counts.append(cplx.counts_)

    results = []

    for n in range(len(ncutoffs)):
        # count_dict = dict.fromkeys(keys, 0.0)
        d = OrderedDict()
        d = d.fromkeys(keys, 0.0)
        # now sort the atom-pairs and accumulate the element-type to a dict
        for e_e, c in zip(rec_lig_element_combines, onion_counts[n]):
            d[e_e] += c

        results += d.values()

    return results, keys


def genfeat_mp(args):
    f, l, nc = args
    r, e = generate_features(f, l, nc)
    print(f)
    return r, e

if __name__ == "__main__":

    print("Start Now ... ")

    d = """
    Predicting protein-ligand binding affinities (pKa) with OnionNet model.
    Citation: Zheng L, Fan J, Mu Y. arXiv preprint arXiv:1906.02418, 2019.
    Author: Liangzhen Zheng (zhenglz@outlook.com)

    This script is used to generate inter-molecular element-type specific
    contact features. Installation instructions should be refered to
    https://github.com/zhenglz/onionnet

    Examples:
    Show help information
    python generate_features.py -h

    Run the script
    python generate_features.py -inp input_samples.dat -out features_samples.csv

    # tutorial example
    cd tuttorials/PDB_samples
    python ../../generate_features.py -inp input_PDB_testing.dat -out PDB_testing_features.csv

    """

    parser = argparse.ArgumentParser(description=d, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("-inp", type=str, default="input.dat",
                        help="Input. The input file containg the file path of each \n"
                             "of the protein-ligand complexes files (in pdb format.)\n"
                             "There should be only 1 column, each row or line containing\n"
                             "the input file path, relative or absolute path.")
    parser.add_argument("-out", type=str, default="output.csv",
                        help="Output. Default is output.csv \n"
                             "The output file name containing the features, each sample\n"
                             "per row. ")
    parser.add_argument("-lig", type=str, default="LIG",
                        help="Input, optional. Default is LIG. \n"
                             "The ligand molecule residue name (code, 3 characters) in the \n"
                             "complex pdb file. ")
    parser.add_argument("-nt", type=int, default=1, help="Input, optional. Default is 1. Use how many of cpu cores.")


    args = parser.parse_args()

    if len(sys.argv) < 2:
        parser.print_help()
        sys.exit(0)

    keys = ["_".join(x) for x in list(itertools.product(ALL_ELEMENTS, ALL_ELEMENTS))]

    with open(args.inp) as lines:
        lines = [x for x in lines if ("#" not in x and len(x.split()) >= 1)].copy()
        inputs = [x.split()[0] for x in lines]
    # defining the shell structures ... (do not change)
    n_shells = 60
    n_cutoffs = np.linspace(0.1, 3.1, n_shells)

    results = []
    ele_pairs = []

    # computing the features now ...
    l = len(inputs)
    lig_code = args.lig
    if args.nt <= 1:
        for i, fn in enumerate(inputs):
            # the main function for featurization ...
            r, ele_pairs = generate_features(fn, lig_code, n_cutoffs)
            results.append(r)
            # success.append(1.)
            print(fn, i, l)
    else:
        pool = mp.Pool(args.nt)
        _args = list(zip(inputs, l * [lig_code,], l*[n_cutoffs]))
        results_all = pool.map(genfeat_mp, _args)
        results = [ x[0] for x in results_all]

    # saving features to a file now ...
    df = pd.DataFrame(results)
    try:
        df.index = inputs
    except IndexError:
        df.index = np.arange(df.shape[0])

    col_n = []
    for i, n in enumerate(keys * len(n_cutoffs)):
        col_n.append(n + "_" + str(i))
    df.columns = col_n
    df.to_csv(args.out, sep=",", float_format="%.1f", index=True)

    print("Feature extraction completed ...... ")

