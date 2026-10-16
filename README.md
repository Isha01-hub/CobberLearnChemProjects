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
# Chapter 7 AlkaneAnalysis 
I used the alkane dataset provided for Chapter 7 to study missing values and compare predictions models in PyCharm. Removing every incomplete row left 100 of the original 250 alkanes, so 60% of the dataset was lost. I compared the molecule size before and after deletion and tried mean imputation as a basic way to fill missing numerical values.
I then tested log-linear regression, K-nearest neighbors, random forests, and an averaged ensemble for viscosity, heat capacity, and thermal conductivity. Log-linear regression had the lowest test MAE for viscosity and thermal conductivity, whole the depth-15 random forest performed best for heat capacity. I also counted predictions with relative errors greater than 100%, since an average error can hide large individual mistakes. The code is in 'Chapter 7 AlkaneAnalysis/AlkaneAnalysis.py', and the source file is 'alkane_dataset.csv.'
## Chapter 7 — CobberImpute Concepts: Python Alkane Analysis
I used the alkane dataset in PyCharm to explore missing values. Removing incomplete rows left only 100 of 250 records, so 60% of the data was lost. I tried mean imputation, then compared log-linear regression, KNN, random forests, and an ensemble using MAE.
Log-linear regression performed best for viscosity and thermal conductivity, while the depth-15 random forest performed best for heat capacity on my test split. I used these models to fill the missing values for those three properties and saved the results, imputation flags, and a correlation heatmap. This helped me understand why deleting data can be costly and why estimated values should be clearly marked.
My code and outputs are in Chapter 7 AlkaneAnalysis. This is my Python analysis of the CobberImpute concepts, separate from MakingDataWhole.
## Chapter 8 - Error Metrics
I completed the CobberLearnChem activity for Chapter 8, but realized I had forgotten to complete the separate Error Metrics assignment. I went back and completed it in PyCharm, calculating MAE, MSE, and R² with NumPy and checking them against scikit-learn. The results matched. I created predicted-versus-actual and residual plots, added a readable values table, and highlighted the largest errors.
I also compared the four Chapter 8 datasets in Python. Low scatter performed best, while high scatter had the largest errors. The deviation plots helped me understand that a high R² can still hide systematic prediction errors. My code, datasets, metrics table, and plots are saved on GitHub in the Error Metrics folder.
## Chapter 9 - DecisionTreeClassifier 
I completed the Python solubility project using the 12 molecules provided in the workbook. I trained a decision tree using molecular weight, hydrogen bond donors, and hydrogen bond acceptors, then saved a labeled tree image.
I added colored nodes and readable text, and compared the original tree with one limited to depth 2. Training accuracy dropped from 100% to 83.3%, showing how a simpler tree can miss patterns. These accuracies describe the training data, not new molecules. My code and both tree images are in DecisionTreeClassifier.
## Chapter 10 - ElementClustering
I collected Group 1 atomic radii and first ionization energies from the Royal Society of Chemistry and saved the data with source links. I plotted the trend, standardized the two features, and used KMeans to form clusters with black Xs marking their centers. I also added a loop to try different cluster counts.
With two clusters, lithium and sodium grouped together, while the other four alkali metals formed the second group. With three clusters, the groups were Li–Na, K–Rb, and Cs–Fr. I repeated the analysis with Group 2, where beryllium and magnesium grouped separately from calcium, strontium, barium, and radium. This helped me see how the selected properties and cluster count affect the grouping.
## RESEARCH PROJECT: ESTROGEN_ANTAGONIST
1. EXPLAIN THE BLA ASSAY MECHANISM
- BLA stands for beta-lactamase, which is an enzyme used as a reporter to measure estrogen receptor activity. In the estrogen receptor alpha BLA assay, researchers use genetically modified HEK293 cells. These cells contain the estrogen-binding portion of the receptor attached to a GAL4 DNA-binding domain. When this system is activated, it turns on a reporter gene that produces beta-lactamase.
Beta-lactamase cuts a fluorescent substrate, changing its signal from green toward blue. The assay measures the ratio of blue to green fluorescence. In antagonist mode, the cells are exposed to estrogen and a test chemical together. If the chemical interferes with estrogen-induced receptor activity, the cells produce less beta-lactamase, which lowers the fluorescence ratio compared with estrogen alone.
A separate viability measurement helps determine whether the reduced signal could be caused by cell damage. This is important because a chemical that harms the cells could lower the reporter signal without specifically blocking the estrogen receptor.

2. WHAT PART OF THE PATHWAY DOES AN ACTIVE MOLECULE DISRUPT?
   - The intended point of disruption is estrogen receptor activation and the gene transcription that follows. A true antagonist can interact with the receptors' ligand-binding region and prevent estrogen from fully activating it. In the BLA assay, this reduces reporter gene activity and lowers the fluorescence ratio. The assay therefore looks at interference with the cell’s response to estrogen rather than interference with estrogen production.
   - However, an active result does not prove that the chemical directly binds to the estrogen receptor. Chemicals can also interfere with the reporter system or the fluorescence measurement. Additional testing is needed to confirm the exact mechanism. In our dataset, the chemicals were classified as active in the BLA ratio assay and inactive in the paired viability assay. This makes them candidates for further investigation rather than confirmed estrogen antagonists.
   3.  How does the assay apply to women’s health?
    - This assay is relevant to women’s health because estrogen signaling is important for reproductive function and hormone-responsive tissues. Screening chemicals for interference with estrogen receptors can help researchers identify substances that need further study for possible effects on fertility, pregnancy, and reproductive health.
    - The assay is also useful in breast cancer research. Some breast cancers depend on estrogen signaling for growth, so blocking that signaling can be helpful. For example, tamoxifen blocks estrogen’s effects in breast tissue. Studying estrogen antagonists can therefore support both environmental health research and drug discovery. However, activity in this assay alone does not show that a chemical is harmful to women or that it would be a safe cancer treatment.
   4. Explain what each column in the dataset represents.
  - The dataset contains chemical identification information, chemical structure information, and assay results. The suffix BLA_Ratio identifies the receptor-related reporter endpoint, while BLA_Viability identifies the paired cell viability endpoint. Columns with the same measurement name are explained together below:
i. DTXSID
A unique substance identifier used in EPA’s chemical database. It helps match the same substance across datasets.
ii. PREFERRED NAME
The chemical name selected by the source database.
iii. CASRN
The chemical’s CAS Registry Number, which is used to identify the substance.
iv. MOLECULAR FORMULA
Shows the elements and number of each type of atom in the chemical. It does not show how the atoms are connected.
v.MONOISOTOPIC MASS
The calculated molecular mass using the principal isotope of each element, usually expressed in daltons.
vi. SMILES
A text representation of the chemical’s molecular structure, including atom connections and bond types.
vii. MORGAN_FINGERPRINT
A computer-generated summary of molecular structure. In this dataset, each fingerprint contains 2,048 zeros and ones. These fingerprints can be used to compare structures or build prediction models.
viii. TOXCAST ACTIVE_BLA_Ratio and TOXCAST ACTIVE_BLA_Viability
These appear to represent the overall number of ToxCast assay endpoints in which the chemical was classified as active. The values are identical between the two versions, suggesting that they are chemical-level summaries copied from the source files.
ix. TOXCAST TOTAL_BLA_Ratio and TOXCAST TOTAL_BLA_Viability
These appear to represent the total number of ToxCast assay endpoints with results for the chemical. They are also identical between the two versions.
x. % TOXCAST ACTIVE_BLA_Ratio and % TOXCAST ACTIVE_BLA_Viability
The percentage of the chemical’s ToxCast endpoints classified as active, calculated approximately as active endpoints divided by total endpoints, multiplied by 100. This is not the percentage of estrogen receptor inhibition.
xi. HIT CALL_BLA_Ratio and HIT CALL_BLA_Viability
The final activity classification for each endpoint: Active or Inactive. All 792 chemicals in this dataset are Active in the ratio endpoint and Inactive in the viability endpoint.
xii. CONTINUOUS HIT CALL_BLA_Ratio and CONTINUOUS HIT CALL_BLA_Viability
Numerical scores that support the activity classification using the concentration–response data and model fit. They do not measure percent inhibition. Some viability scores are negative, so their exact interpretation depends on the source’s scoring convention.
xiii. TOP_BLA_Ratio and TOP_BLA_Viability
The modeled response amplitude from the best-fitting concentration–response curve. Negative values can represent a decrease in the measured signal relative to the reference. TOP is a response measurement, not a concentration.
xiv. SCALED TOP_BLA_Ratio and SCALED TOP_BLA_Viability
The modeled response scaled relative to the assay’s activity cutoff. This makes results from endpoints with different response scales easier to compare.
xv. AC50_BLA_Ratio and AC50_BLA_Viability
The estimated concentration producing 50% of the modeled maximal response. For reliable active responses within the same assay, a lower AC50 generally indicates greater potency. ToxCast commonly reports these values in micromolar units, but the original export should be checked to confirm the units.
xvi. LOGAC50_BLA_Ratio and LOGAC50_BLA_Viability
The base-10 logarithm of AC50. This puts concentrations on a scale that is easier to compare and graph. For example, an AC50 of 10 corresponds to a LOGAC50 of 1.
-  EPA's documentation describes TOP and scaled responses measurements and on the other hand the ToxCast pipeline publication explains the continuous hit calls and AC50 values. The combined dataset contains 792 chemical records, but only 396 have SMILES and Morgan Fingerprints. Also, the viability AC50 values should not be treated as confirmed toxicity concentrations because the viability responses were classified as inactive. Potency estimates from inactive curves are generally not informative. 
