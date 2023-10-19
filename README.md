# TREKER 

We present **TREK** (**T**oward Document-Level **R**elation **E**xtraction in **K**orean) dataset constructed from Korean encyclopedia documents written by the domain experts. 

We implement our Korean document-level RE model based on **TREK** dataset considering named entity-type information and Korean language characteristics, and entitled as **TREKER**.


## Requirements

```
PyTorch = 1.9.0
HuggingFace Transformers = 4.8.1
dill
opt_einsum
wandb
```

## TREK Dataset
Our **TREK** dataset can be downloaded <a href="https://drive.google.com/file/d/11VX3xwhjVknDw0iqitpt1g4XbHWkBgwn/view?usp=sharing">here</a>.

(*Due to legal review issues with NAVER corp., only a portion of the dataset is publicly available. We will release the full dataset soon.*)

## TREKER 
### Run

Given an input dataset:

1. The ```dataset/TREK_dataset/Original``` folder contains the original files provided by the corresponding dataset that are necessary for our experiments.
1. The command ```bash Code/prepare.sh``` transforms the original data structure into the structure acceptable to our model and stores the output files in the ```dataset/TREK_dataset/Processed``` folder.
1. The command ```bash Code/main.sh``` trains the model, writes the standard output in the ```dataset/TREK_dataset/Stdout``` folder, and delivers the set of predicted relations and corresponding evidence for the develop and test sets in the ```dataset/TREK_dataset/Processed``` folder.

The set of hyperparameters for Step 2 and 3 are specified in ```prepare.sh``` and ```main.sh```, respectively. 

