Ensemble Deep Learning Models on Raw DNA Sequences for Viral Genome Identifi cation in Human Samples

# ViralMiner - A Tool for Viral Genome Identification in Metagenomic Data

ViralMiner is a tool designed to identify and analyze viral sequences from metagenomic data. It provides a comprehensive pipeline for the detection of viral genomes, enabling researchers to explore the diversity and function of viruses in various environments.

## Installation / Requirements

No specific version of Python/PyTorch is required.
However, we used Python 3.10 and these libraries:

```bash

torch==2.3.1+cu118
torchvision==0.18.1+cu118
pandas == 2.3.3
numpy==2.2.6
scikit-learn==1.7.2
```

## Usage

### Training

There are two types of models to train: Branch and Merger.
A Merger is a model that takes the output of the Branches and produces a final prediction.
We defined 3 types of Branch models: Frequency, Pattern, and LP. Each Branch model is trained separately, and then the Merger model is trained on the output of the Branch models.

We provide training script files for the best models that we found (see the paper).


## Reproduce Paper Results (Ensemble)

To reproduce the results of the paper, you can use the following command:

```python
python scripts/produce_logits_all.py [PATH to dataset.csv]
# this will produce logits.csv for each model that we used in the paper's ensemble.
# by not providing the save_folder argument, the logits will be saved in the same folder as the model.ptm file, in the form of a logits.csv file.

python scripts/test_ensemble.py
# this will load all the logits.csv files produced in the previous step.
# the script will then produce the ensemble predictions and evaluate the performance of the ensemble model, printing the results to the console.

## you should obtain the same results as in the paper:
[i] auroc: 0.939085556097011
```

---

## Dataset description

In ```dataset/``` folder, you can find two datasets:
```
dataset/DNA_data.rar
dataset/noise.rar
dataset/UnseenVirus.rar
```

The first, ```DNA_data.rar```, contains the raw DNA sequences that were used for training and testing the models.
In the paper this dataset is referred to as "VM"
<br>
(Tampuu A, Bzhalava Z, Dillner J, Vicente R (2019) ViraMiner: Deep learning on raw DNA sequences for identifying viral genomes in human samples. PLOS ONE 14(9): e0222271. https://doi.org/10.1371/journal.pone.0222271)

The second, ```noise.rar```, contains simulated noisy data that can be used for testing the models. We generated it from VM test set by introducing random point mutations (substitutions) at different rates. For each mutation rate, we created five distinct versions of the test set using different seeds for the pseudo-random number generator, enabling statistical analysis.<br>
Specifically:
 - 3 levels of noise: low (1%), medium (5%), and high (10%).
 - for each level of noise we generated 5 different datasets, each with a different random seed (for a total of 15 datasets).


The third, is the "unseen" dataset. Training and validation datasets are derived from the VM dataset, with all anellovirus 
sequences removed from both sets. The test set is composed of both non-viral sequences 
(26,296 contigs) and anellovirus sequences (1,348 contigs). In this setting, anelloviruses 
represent a completely novel viral category from the model’s perspective, as they are not 
encountered during training. This setup enables evaluation of the model’s capability to 
identify viruses belonging to entirely unseen classes. 

---


### Training a single model

```python
python scripts/train.py --d ../dataset/ -p [insert PATH] --save_dir [insert PATH]

# for example for a branch:
python scripts/train.py --d ../dataset/ -p ready_to_train_files/final/onehot/branch/frequency/init+norm.json --save_dir model_zoo_local/branch/frequency/init+norm.json

# for example for a merger
python scripts/train.py --d ../dataset/ -p ready_to_train_files/final/onehot/merger/frequency+pattern+lp/init+norm.json --save_dir model_zoo_local/merger/frequency+pattern+lp/init+norm.json
```

---


### Testing a single model

To evaluate the performance of ViralMiner, you can use the following command:

```python
python scripts/test.py data/fullset_test.csv -m model_zoo/onehot/merger/frequency+pattern+lp/init+norm/model.ptm
```

### Producing logits for a single model
To produce logits (that will be saved as .csv) use the following command:
```python
python scripts/produce_logits.py dataset/fullset_test.csv -m model_zoo/onehot/merger/frequency+pattern+lp/init+norm/model.ptm --save_folder frequency+pattern+lp-init+norm.csv
```

---

## Citation
If you use ViralMiner in your research, please cite the following paper:

```
TBD (Paper Under Review in MDPI Sensors Journal)
```
