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
```


### Training a single model

```python
python scripts/train.py --d ../dataset/ -p [insert PATH] --save_dir [insert PATH]

# for example for a branch:
python scripts/train.py --d ../dataset/ -p ready_to_train_files/final/onehot/branch/frequency/init+norm.json --save_dir model_zoo_local/branch/frequency/init+norm.json

# for example for a merger
python scripts/train.py --d ../dataset/ -p ready_to_train_files/final/onehot/merger/frequency+pattern+lp/init+norm.json --save_dir model_zoo_local/merger/frequency+pattern+lp/init+norm.json
```

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


## Citation
If you use ViralMiner in your research, please cite the following paper:

```
TBD
```