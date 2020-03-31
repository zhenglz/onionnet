#!/bin/bash

# Author: Liangzhen Zheng
# Date: Mar 31, 2020 
# Version: 0.1

# put ligandnamechanger.py in your ~/bin or PATH
# cp ligandnamechanger.py ~/bin

# Usage
if [ $# -ne 3 ]; then
    echo "usage: "
    echo "bash prepare_complex.sh receptor ligand output "
    echo "example: "
    echo "bash prepare_complex.sh receptor.pdb docked_ligand.pdb output_complex.pdb"
    exit 0;
fi

# receptor
rec=$1
lig=$2
out=$3

# 1. change ligand name for ligand first
ligandnamechanger.py $lig ${lig}_temp.pdb

# 2. concatenate the receptor and ligand
cat $rec ${lig}_temp.pdb | awk '$1 ~ /ATOM/ || $1 ~ /HETATM/ {print $0}' > $out

