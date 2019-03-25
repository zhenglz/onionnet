#!/usr/bin/env python

import sys
import dockml.pdbIO as pdbio
import os

def lig_name_change(lig_in, lig_out, lig_code):

    pio = pdbio.rewritePDB(lig_in)
    tofile = open(lig_out, "w")
    with open(lig_in) as lines:
        for s in lines:
            if len(s.split()) and s.split()[0] in ['ATOM', 'HETATM']:
                nl = pio.resNameChanger(s, lig_code)
                #n2 = pio.chainIDChanger(nl, "Z")
                tofile.write(nl)

    tofile.close()
    return None

def main():
    if len(sys.argv) <= 3:
        print("""Usage: \npython ligandnamechanger.py old_ligand.pdb new_ligand.pdb""")

    lig = sys.argv[1]
    out = sys.argv[2]

    with open(lig) as lines:
        lines = [x for x in lines if "LIG" in x]
        if not len(lines):
            lig_name_change(lig, out, "LIG")

        else:
            os.system("cp %s temp"%(lig))

main()

