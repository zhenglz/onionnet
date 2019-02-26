# OnionNet
A multiple-layer inter-molecular contact features based deep neural network for protein-ligand binding affinity prediction

# Citation


# Installation
Necessary packages should be installed to run the OnionNet model.

Dependecies:

    python 
    numpy  
    scipy  
    pandas 
    scikit-learn
    mpi4py
    mdtraj 
    tensorflow


To install necessary environment, create a new env with conda commands
   
    conda env create -f onionnet_environment.yml 
    source activate onionnet
    # do something yourself

# Usage
## 1. Prepare the protein-ligand complexes (3D structures) in pdb format
a. The protein-ligand complexes from experimental crystal or NMR structures, or from molecular
docking, are accepted.
b. Make sure that the residue name of the ligands is the same, preferable "LIG" or "UNK".
c. Generate an file containing the complexes, one complex per line. Each line contains the 
path of the protein-ligand complex file.

## 2. Generate multiple-layer inter-molecular contact features
Using the "generate_features.py" script to generate the features for OnionNet predictions.
 

    python generate_features.py input_complexes.dat output_features.py
    # or run the script with MPI
    mpirun -np 4 python generate_features.py input_complexes.dat output_features.py 

## 3. Predict the pKa of the complexes
Given a dataset containing the multiple-layer inter-molecular contact features, we could predict
the binding affinities (in pKa scale). 
        
    python predict_pKa_HFree.py OnionNet_HFree.h5 output_predicted_pKa.csv


