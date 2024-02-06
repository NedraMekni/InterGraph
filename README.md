InterGraph
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/NedraMekni/InterGraph/workflows/CI/badge.svg)](https://github.com/NedraMekni/InterGraph/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/NedraMekni/InterGraph/branch/master/graph/badge.svg)](https://codecov.io/gh/NedraMekni/InterGraph/branch/master)


Welcome to our package for transforming Protein-Ligand Complex PDB files into multigraph representations! This package allows you to analyze protein-ligand interactions by constructing a multigraph representation based on user-defined distance thresholds from the ligand atoms.

## Installation
Install InterGraph master via ```git clone``` and ```python setup install```:
```
git clone https://github.com/NedraMekni/InterGraph.git
cd InterGraph
python setup.py install
```

### Package Functionality

The package provides a comprehensive pipeline for modeling protein-ligand interactions and predicting binding affinity :

- Clean the PDB file by removing salts, ions, metals, and water molecules.
- Select the relevant chain containing the ligand in the case of multichain complexes.
- Construct a multigraph representation of the protein-ligand complex based on user-defined distance thresholds.
- Train a Graph Convolutional Network (GCN) using the generated PDB files to predict binding affinity.
- Export the trained GCN model for future use.

### Notebook

For a detailed demonstration of the package usage and analysis workflow, refer to our Jupyter Notebook:

[Link to Notebook](https://github.com/NedraMekni/multigraph-colab/blob/main/multi_g_generation.ipynb)


## Input File Description

### Overview

The input file for our package is a Protein Data Bank (PDB) file representing a protein-ligand complex. This file serves as the basis for creating a multigraph representation of the complex. Subsequently, the generated PDB files are utilized to train a Graph Convolutional Network (GCN) for the prediction of binding affinity.

### File Format

The input file is expected to be in PDB format, a widely used file format for representing biomolecular structures. 

### Data Directory

The input PDB files are stored in the data directory within the package.

### File Preprocessing

Before constructing the multigraph representation, the input PDB file undergoes preprocessing steps:

1. **Removal of Salts, Ions, and Metals:** Any salts, ions, or metal atoms present in the PDB file are removed
   
3. **Removal of Water Molecules:** Water molecules are removed from the PDB file during preprocessing.

4. **Selection of Relevant Chain:** In the case of multichain complexes, only the chain containing the ligand is retained for further analysis. 

### Distance Thresholds

The multigraph representation is constructed based on three distance thresholds from the ligand atoms. These thresholds define the proximity at which protein atoms interact with the ligand atoms and influence the connectivity of the multigraph.

### Model Training

After generating the multigraph representations, the PDB files are used to train a Graph Convolutional Network (GCN) for the prediction of binding affinity. 

### Exporting Trained Model

The trained GCN model can be exported from the package using the `model_trained` function. This allows users to save the trained model for future use or deployment in other applications.



## Customization
You can customize the distance thresholds used to build the multigraph representation. Simply modify the distance_thresholds list in the create_multigraph function call to suit your specific needs.


### Copyright

Copyright (c) 2022, Nedra Mekni


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
