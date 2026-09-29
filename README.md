## AlkaneBoilingPoints
This simple code plots the number  of carbon atoms in the x-axis and the boiling points of that  alkane on the y-axis. It generates a properly labeled graph using Matplotlib.
This is part of the CobberLearnChem Machine Learning course through Concordia College.
Alongside my public coding projects, I'm keeping a private ethics portfolio where I reflect on what I'm learning and how it's shaping the kind of scientist I want to become.
## PubChemFetcher
This project uses the PubChemPy library to retrieve chemical information about theobromine from PubChem. The script displays its PubChem CID, molecular formula, molecular weight, and SMILES string. I also improved the output formatting to make the chemical information easier to read.
# MoleculeExplorer
This project uses RDkit to analyze molecules from their SMILES strings and calculate molecular descriptors. The program calculates exact molecular weight, hydrogen bond donors, hydrogen bond acceptors, TPSA, and number of rings. I tested the program with ethanol, acetic acid, and caffeine.
Through this project, I learned how SMILES can be converted into molecule objects and how different molecular descriptors can provide different information about a molecule's properties and structure.
# MakingDataWhole
This chapter 7 project explores missing ages in the Titanic dataset using Python in Pycharm. I first filled missing ages with the mean, then examined a correlation heatmap. Passenger class and number of siblings or spouses had the strongest numeric correlations with age in this dataset. 
I tested K-nearest neighbors (KNN) by hiding the ages of 143 passengers whose ages were known. It's mean absolute error was "9.96 years". A random forest tested on the same passengers had a slightly lower error of "9.70". I then used KNN to estimate the 177 originally missing ages. The mean age changed from "29.70" to "29.59 years", and no ages remained missing. 
The code saves a prediction plot, a correlation heatmap, and a completed CSV. The CSV marks which ages were estimated. These predictions are useful for analysis but should not be treated as a measured passenger ages.
