# OnionNet
A multiple-layer inter-molecular contact based deep neural network for protein-ligand binding affinity prediction.
The testing set is CASF-2013 benchmark and PDBbind v2016 coreset. The protein-ligand binding affinity is directly predicted.

The model could be applied for re-scoring the docking results.

<img src="./datasets/TOC.png" alt="CNN aided protein-ligand binding affinity prediction and docking rescoring">


## Contact
<p>Yuguang Mu, Nanyang Technological University, ygmu_AT_ntu.edu.sg</p>
<p>Liangzhen Zheng, Nanyang Technological University, lzheng002_AT_e.ntu.edu.sg</p>


## Citation
<a href='https://arxiv.org/abs/1906.02418'>Zheng L, Fan J, Mu Y. OnionNet: a multiple-layer inter-molecular contact based convolutional
neural network for protein-ligand binding affinity prediction[J]. arXiv preprint arXiv:1906.02418, 2019. </a>


## Installation
Necessary packages should be installed to run the OnionNet model.

Dependecies:

    python >= 3.6
    numpy  
    scipy  
    pandas 
    scikit-learn
    mdtraj 
    tensorflow


To install necessary environment, create a new env with conda commands
   
    # download the package and then enter the folder
    git clone https://github.com/zhenglz/onionnet.git
    cd onionnet

    # create a new pearsonal conda environment
    conda create -n onionnet python=3.6
    conda activate onionnet

    # install necessary packages
    conda install -c anaconda scipy numpy pandas
    conda install tensorflow
    conda install -c omnia mdtraj
    
    # do some tests now
    python generate_features.py -h
    python predict_pKa.py -h


## Usage
### 1. Prepare the protein-ligand complexes (3D structures) in pdb format
    
    a. The protein-ligand complexes from experimental crystal or NMR structures, or from molecular
       docking, are accepted.
    b. Make sure that the residue name of the ligands is the same, preferable "LIG" or "UNK".
    c. Generate an file containing the complexes, one complex per line. Each line contains the 
       path of the protein-ligand complex file.

### 2. Generate multiple-layer inter-molecular contact features
Using the "generate_features.py" script to generate the features for OnionNet predictions.
 
    python generate_features.py -h
    python generate_features.py -inp input_complexes.dat -out output_features.csv

The input file contains the absolute or the path of the protein-ligand complexes pdb files.
The content of the "input_complexes.dat" file could be:
 
    ./10gs/10gs_complex.pdb
    ./1a28/1a28_complex.pdb

Or:
  
    /home/liangzhen/PDBBind_v2018/10gs/10gs_dockingpose.pdb
    /home/liangzhen/PDBBind_v2018/1a28/1a28_dockingpose.pdb


Note: make sure you only cat one receptor with one docking pose into a complex file.


### 3. Predict the pKa of the complexes
Given a dataset containing the multiple-layer inter-molecular contact features, we could predict
the binding affinities (in pKa scale). 

    python predict_pKa.py -h
    python predict_pKa.py -model OnionNet_HFree.model -scaler StandardScaler.model -fn features.csv -out output_predicted_pKa.csv


