# CRAF-Structure-Based-Drug-Generation
Contrastive Representation Learning Enhanced Autoregressive Flows for Structure-Based Drug Generation

## Create the Conda Environment
```
conda env create -f environment.yml
conda activate craf_env
```

## Molecular Generation
To use the model, please run the following command:
```
python main_generate.py --config main_generate_config.yaml
```
### Configuration
The `main_generate_config.yaml` file contains essential parameters for inference. Before running, please ensure the following paths and settings are correctly configured:

* `pocket`: the pdb file of pocket in receptor.
* `ckpt`: the path of saved model (`.pt` file).
* `num_gen`: the number of generative molecule.
* `name`: receptor name.
* `device`: The computing device to use (e.g., `cuda 0` for GPU or `CPU`.)

### Receptor Preparation
To extract the pocket structure from your receptor protein, run `create_pocket_pdb.py`. This script requires the receptor protein (`.pdb`) and its co-crystallized ligand (`.sdf`) to define the binding site. By default, the script extracts residues within 10Å of the reference ligand.
```
python create_pocket_pdb.py --protein /your/protein/file.pdb --ligand /its/ligand/file.sdf
```
## Data Preparation
The pre-training and fine-tuning datasets are hosted on Zenodo: `https://zenodo.org/records/18898386`.  
To prepare the data for training, follow these steps:  
**Download and Organize**: Download the datasets and place them into the `dataset/` folder.  
**Pre-processing**: Run the following script to process ligands from the ZINC database:  
```
python make_ZINC_pretrain_data.py
```
This will generate an `LMDB` file in the `pretrain_data/` directory.

## Model Training
The CRAF model training involves a three-stage pipeline. The **Fine-tuning** stage leverages the weights from both **Contrastive** and **Ligand** pre-training stages for initialization.

**Stage 1: Contrastive Learning Pre-training**  
This stage focuses on learning robust representations through contrastive objectives. First, construct the molecular graphs, then start the training:  
```
python make_graph.py
python contrastive_train.py --config contrastive_train_config.yml
```
* Output: The pre-trained checkpoint will be saved in the`cl_pretraining_log` directory.

**Stage 2: Ligand Pretraining**  
Refine the generative flow's understanding of molecular distributions:  
```
python ligand_pretraining.py
```
* Output: The pre-trained checkpoint will be saved in the `ligand_pretraining_log` directory.

**Stage 3: Finetuning**  
In this final stage, the model is initialized with the weights from the two pre-training modules (Stage 1 & 2) and optimized for structure-based drug generation:
```
python finetuning.py
```
* Output: The final model checkpoint will be saved in the `finetuning_log` directory.

## Citation
TODO


