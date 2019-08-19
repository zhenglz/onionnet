#!/bin/bash

# get a list of vina out file list
for pdbqt in ./*.pdbqt
do
    # split the vinaout output poses into separated files
    obabel $pdbqt -O ${pdbqt}_.pdb -m

    # by default, only cat the first vina out pose with the receptor
    cat ./receptor.pdb ${pdbqt}_0.pdb | awk '$1 ~ /ATOM/ {print $0}' > ${pdbqt}.complex.pdb

    echo "Complete " $pdbqt

done

# prepare an input file for the feature generation tool
ll -rt ./*.complex.pdb > docked_complexes.list
