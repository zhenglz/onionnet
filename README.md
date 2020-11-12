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
    # Git Large File System usage: https://www.atlassian.com/git/tutorials/git-lfs   
    git lfs clone https://github.com/zhenglz/onionnet.git
    cd onionnet

    # create a new pearsonal conda environment
    conda create -n onionnet python=3.6
    conda activate onionnet

    # install necessary packages
    conda install -c anaconda scipy numpy pandas
    conda install tensorflow
    conda install -c omnia mdtraj
    conda install -c openbabel openbabel
    
    # do some tests now
    python generate_features.py -h
    python predict.py -h

Or alternatively, install the packages through the environment file.

    # create a new conda environment (name: onionnet)
    conda env create -f onet_env.yaml
    conda activate onionnet
    

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


Note: make sure you cat one receptor with one docking pose into a complex file.

However, in some situations, we have protein and ligand in separated files. To generate features with protein-ligand pair with protein in PDB format and ligand in mol2 format, please refer to this repo:

    https://github.com/zhenglz/onionnet_featurize


### 3. Predict the pKa of the complexes
Given a dataset containing the multiple-layer inter-molecular contact features, we could predict
the binding affinities (in pKa scale). 
An example dataset file could be found in ./datasets  

    python predict.py -h
    python predict.py -fn datasets/features_testing_v2016core_290_pka.csv -scaler models/StandardScaler.model -weights models/CNN_final_model_weights.h5 -out datasets/output_v2016_predicted.csv

    # tutorial example
    cd tutorials/PDB_samples
    # generate features
    python ../../generate_features.py -inp input_PDB_testing.dat -out PDB_testing_features.csv
    # predict binding affinities 
    python ../../predict.py -fn  PDB_testing_features.csv -out predicted_pKa.csv -weights ../../models/CNN_final_model_weights.h5 -scaler ../../models/StandardScaler.model

### 4. FAQ
#### a. "ValueError: PDB Error: All MODELs must contain the same number of ATOMs"
This issue comes from the fact that in the PDB parsing process, the ligand atoms have not been correctly identified. The PDB parsing process trys to extract the xyz coordinates and element information from both the receptor (generally a protein) and the ligand. 
A package (mdtraj) is used to perform the parsing, based on the key words: protein (for receptor) and LIG (for ligand). 
Thus you should make sure that the residue name (in lines starting with ATOM, col 18-20) of the ligand atoms in a PDB file is LIG (default). To achieve this, you may use a shell script prepare_complex.pdb (in tools/) to do this, or use a text editor.  

example: 
```bash prepare_complex.sh 10gs_protein.pdb 10gs_ligand.mol2.pdb 10gs_complex.pdb```

#### b. How to convert ligand file to PDB format?
You could use openbabel for the format converting. Openbabel could automately convert the molecule into your desired format based on the extension of your output file name. You may use conda to install openbabel:

```
# install
conda install -c openbabel openbabel
# usage example
obabel 10gs_ligand.mol2 -O 10gs_ligand.mol2.pdb
# or
obabel 10gs_ligand.mol2 -O 10gs_ligand.pdb
```

#### c. The downloaded CNN model file size seems not correct?
Github has restrictions for single-large size file, and band-width usage. Thus, from time to time, the CNN model file in models/ folder could not be correctly downloaded, and I couldn't find a good solution to this issue yet. Please delete the file manually, and then donwload the CNN model file (around 550MB) from the following onedrive link:
```
https://drive.google.com/file/d/1cwJN44TgaVBWYEEb_SGU5JBJp6WbFdM1/view?usp=sharing
```

Or download the model file with the following command to replace the file in the model directory:

    wget "https://drive.google.com/uc?export=download&id=1cwJN44TgaVBWYEEb_SGU5JBJp6WbFdM1" -O "CNN_final_model_weights.h5"
