# This program uses PubChemPy to retrieve and display chemical information about theobromine.

import pubchempy as pcp

compound = pcp.Compound.from_cid(5429)

print("\nTheobromine Chemical Information")
print("--------------------------------")
print("PubChem CID:       5429")
print("Molecular Formula:", compound.molecular_formula)
print("Molecular Weight: ", compound.molecular_weight)
print("SMILES:           ", compound.smiles)