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

if [ ! -f ~/bin/ligandnamechanger.py ] || [ ! -f ligandnamechanger.py ]; then
    echo "Error! No such file ligandnamechanger.py in PWD or ~/bin/; exit now !!!"
    exit 1;
fi

# receptor
rec=$1
lig=$2
out=$3

# 1. change ligand name for ligand first
if [ -f ligandnamechanger.py ]; then
    python ligandnamechanger.py $lig ${lig}_temp.pdb
else
    python ~/bin/ligandnamechanger.py $lig ${lig}_temp.pdb
fi

# 2. concatenate the receptor and ligand
cat $rec ${lig}_temp.pdb | awk '$1 ~ /ATOM/ || $1 ~ /HETATM/ {print $0}' > $out

# 3. do the clean-up
rm -f ${lig}_temp.pdb
