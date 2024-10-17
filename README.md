# ALFABET-LITE: A machine-Learning derived, Fast, Accurate Bond dissociation Enthalpy Tool based on ALFABET

This library contains the trained graph neural network model for the prediction of homolytic bond dissociation energies (BDEs) of organic molecules with C, H, N, and O atoms, which was originally published in St. John et al. (2020) and availalbe [here](https://github.com/NREL/alfabet). This **Lite** version of the model is a lightweigted version of the full model framework focusing on the prediction side only, which has the following advantages:
- Optimized dependencies and compatibility
  - Enhanced compatibility with modern Python and Tensorflow versions
  - Dropped dependency on [nfp](https://github.com/NREL/nfp)
- Dropped unused modules

Some thing you should be aware of:
- Make sure the Python version >= 3.9 and keras version <= 2.15.
- To remain consistency, this repo is still relying on the original alfabet [model repo](https://github.com/pstjohn/alfabet-models), where the model files are published. 
- Apple Silicon (M series chip) users should always check pip/conda builds of the following dependencies whenever encountered issues. Sometimes switching to a different build version would fix the issue.
    - rdkit
    - tensorflow
    - tensorflow-metal


>The model breaks all single, non-cyclic bonds in the input molecules and calculates their bond dissociation energies. Typical prediction errors are less than 1 kcal/mol. 

For additional details, see the publication:
St. John, P. C., Guan, Y., Kim, Y., Kim, S., & Paton, R. S. (2020). Prediction of organic homolytic bond dissociation enthalpies at near chemical accuracy with sub-second computational cost. Nature Communications, 11(1). doi:10.1038/s41467-020-16201-z

## Installation

```bash
$ # Under a virtual environment, clone the repo
$ git clone https://github.com/NREL/alfabet-lite.git
$ # Move into the repo directory
$ cd alfabet-lite
$ # Install the package only, without tensorflow
$ pip install .
$ # If you want to install tensorflow version 2.15 as well
$ pip install alfabet-lite[tensorflow]
$ # If you are using Apple Silicon (M series chip) and want GPU support using tensorflow-metal
$ pip install alfabet-lite[tensorflow,apple_silicon]
```
