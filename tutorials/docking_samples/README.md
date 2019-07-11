# Generate features for PDB or docked protein-ligand complex. 

For each complex file, just place the file path into the input 
file ("input_docked_complexes.dat"), one file per line.

## 1. Make sure that the ligand residue name in each complex file is LIG or UNK. 
Otherwise, error massage or wrong features would generate.

You could use ligandnamechanger.py to change the ligand name.
Usage:

    python ligandnamechanger.py 10gs_vinaout_1.pdb 10gs_vinaout_1_renamed.pdb

## 2. Cat the receptor pdb and the ligand pdb

    cat 10gs_protein.pdb 10gs_vinaout_1_renamed.pdb > 10gs_docked_complex.pdb
    # generate pdb format ligand using open babael
    obabel 10gs_ligand.mol2 10gs_ligand.pdb
    cat 10gs_protein.pdb 10gs_ligand.pdb > 10gs_complex.pdb

## 3. Generate features using generate_features.py
Example commands:
    
    python generate_features.py -h
    python generate_features.py -inp input_PDB_testing.dat -out PDB_testing_features.csv
    python generate_features.py -inp input_docked_complexes.dat -out docking_complexes_features.csv

## 4. Make the prediction

    python predict_pKa.py -h
    python predict_pKa.py -fn docking_complexes_features.csv -model ../../models/OnionNet_HFree.model \
    -scaler ../../models/StandardScaler.model -out predicted_pka_values.csv

Note: The larger the pka value is, the stronger it binds to a receptor.
