# TREKER 

We present **TREK** (**T**oward Document-Level **R**elation **E**xtraction in **K**orean) dataset constructed from Korean encyclopedia documents written by the domain experts. 

We implement our Korean document-level RE model based on **TREK** dataset considering named entity-type information and Korean language characteristics, and entitled as **TREKER**.


## Requirements

```
PyTorch = 1.9.0
HuggingFace Transformers = 4.8.1
```

## Dataset
Our **TREK** dataset can be downloaded here

## Run

Given an input dataset (e.g., _DocRED_):

1. The ```Data/{dataset}/Original``` folder contains the original files provided by the corresponding dataset that are necessary for our experiments.
1. The command ```bash Code/prepare.sh``` transforms the original data structure into the structure acceptable to our model and stores the output files in the ```Data/{dataset}/Processed``` folder.
1. The command ```bash Code/main.sh``` trains the model, writes the standard output in the ```Data/{dataset}/Stdout``` folder, and delivers the set of predicted relations and corresponding evidence for the develop and test sets in the ```Data/{dataset}/Processed``` folder.

The set of hyperparameters for Step 2 and 3 are specified in ```prepare.sh``` and ```main.sh```, respectively. 

Our model trained on _DocRED_ can be downloaded <a href="https://drive.google.com/drive/folders/1_xM8GdK0G5geYn0t4_L4ONOobLHSpznO?usp=sharing">here</a>.


