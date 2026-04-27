# UKAntibody: Antibody CDR-H3 Generation via Partially Latent Flow Matching

A generative model for atomistic antibody design, adapted from the La-Proteina protein generation framework. UKAntibody focuses on CDR-H3 loop scaffolding given antigen and antibody framework context.

## Overview

UKAntibody fine-tunes a partially latent flow matching model on antibody-antigen complex structures from SAbDab. Given a fixed antigen and antibody framework (light/heavy chain), the model generates diverse, designable CDR-H3 loops at full-atom resolution.

## Data

Structural data is sourced from [SAbDab](https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/):

```bash
wget https://opig.stats.ox.ac.uk/webapps/newsabdab/sabdab/archive/all/ -O all_structures.zip
unzip all_structures.zip
```

Preprocessed dataset splits are provided in `dataset/` (`sabdab_all.json`, `rabd_all.json`, etc.).

## Setup

```bash
mamba env create -f environment.yaml
mamba activate laproteina_env
pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu118
pip install graphein==1.7.7 --no-deps
pip install torch_geometric torch_scatter torch_sparse torch_cluster -f https://data.pyg.org/whl/torch-2.7.0+cu118.html
```

Download the pretrained base checkpoints into `./checkpoints_laproteina/`:
- `LD4_motif_idx_aa.ckpt` — indexed all-atom motif scaffolding model
- `AE3_motif.ckpt` — autoencoder for motif scaffolding

## Data Preprocessing

```bash
bash script_utils/run_preprocess_antibody_data.sh
```

This parses SAbDab PDB files and extracts antigen/light/heavy chain coordinates and sequences into the format expected by the dataloader.

## Fine-tuning

```bash
python proteinfoundation/train.py --config-name train_antibody
```

Configuration is in `configs/train_antibody.yaml`. Key parameters:
- `pretrain_ckpt_path`: base checkpoint to fine-tune from
- `opt.lr`: learning rate (default `1e-5`)
- `opt.max_epochs`: number of epochs (default `100`)

## Inference

Prepare the inference input for a target complex:

```bash
python prepare_antibody_inference.py --pdb <path/to/complex.pdb> --config_out <output_config.yaml>
```

Then run generation:

```bash
python proteinfoundation/generate.py --config_name inference_antibody
```

The target task and number of samples are configured in `configs/inference_antibody.yaml`.

## Evaluation

Designability is evaluated using ProteinMPNN. Download weights first:

```bash
bash script_utils/download_pmpnn_weights.sh
```

Then evaluate generated samples:

```bash
python proteinfoundation/evaluate.py --config_name inference_antibody
```

## License

Source code: Apache 2.0. Model weights: [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/).
