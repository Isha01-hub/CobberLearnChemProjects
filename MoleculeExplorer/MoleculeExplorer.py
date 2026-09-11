# This program uses RDKit to calculate molecular descriptors from a SMILES string.

from rdkit import Chem
from rdkit.Chem import Descriptors, Lipinski

smiles = input("Enter a SMILES string: ")

molecule = Chem.MolFromSmiles(smiles)

if molecule is not None:
    print("\nMolecular Descriptor Results")
    print("----------------------------")
    print("Exact Molecular Weight:", Descriptors.ExactMolWt(molecule))
    print("Hydrogen Bond Donors:", Lipinski.NumHDonors(molecule))
    print("Hydrogen Bond Acceptors:", Lipinski.NumHAcceptors(molecule))
    print("TPSA:", Descriptors.TPSA(molecule))
    print("Number of Rings:", Lipinski.RingCount(molecule))
else:
    print("Invalid SMILES string. Please check your input.")