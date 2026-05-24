# FedSHA

Official implementation of the paper "FedSHA: Structure-aware Heterogeneity Adaptation for Personalized Federated Graph Node Classification".

## Overview

FedSHA is a personalized federated graph learning method for node classification under both structural and statistical heterogeneity. The core idea is to perform structure-aware enhancement before personalized optimization, so that adaptive local aggregation is applied on an enhanced local graph view rather than on the original incomplete subgraph.

## Method

FedSHA contains two main components:

- Structural Enhancement (SE): compensates for missing cross-client neighborhood context caused by graph partitioning.
- Adaptive Local Aggregation (ALA): adaptively fuses the global model and the previous local model for client-specific personalization.

This staged design is used to alleviate both partition-induced propagation bias and Non-IID-induced client drift.

## Supported Methods

The implementation supports the following paper-facing method names:

- FedSHA
- FedSHA-1hop
- FedSHA-2hop
- FedSHA w/o SE
- FedSHA w/o PA
- FedGCN
- FedALA-GCN
- FedProx
- FedAvg-GCN

## Datasets

The code supports the following node-classification datasets:

- Cora
- Citeseer
- Amazon-Computers

Dataset files are not tracked in this repository. They are expected to be downloaded or processed locally by PyTorch Geometric.

## Experimental Settings

Unless otherwise specified, experiments use 300 communication rounds, 5 local training steps, 3 GNN layers, a Dirichlet concentration parameter of 0.1, and 2-hop structural enhancement when applicable. Experiments are conducted with client numbers in `{5, 10, 15}` and random seeds in `{1971, 1972, 1999, 2025, 3407}`.

Dataset-specific settings are summarized below.

| Dataset | Learning Rate | Nodes | Feature Dim |
|---|---:|---:|---:|
| Cora | 0.1 | 2708 | 1433 |
| Citeseer | 0.1 | 3327 | 3703 |
| Amazon-Computers | 0.01 | 13752 | 767 |

Method-specific hyperparameters are summarized below.

| Parameter | Value | Used By |
|---|---:|---|
| `eta` | 1 | FedSHA / ALA |
| `rand_percent` | 80 | FedSHA / ALA |
| `layer_idx` | 2 | FedSHA / ALA |
| `prox_mu` | 0.1 | FedProx |

## Repository Structure

```text
.
|-- fedsha.py       # Main implementation
|-- environment.yml # Conda environment file
|-- CITATION.cff
|-- README.md
`-- .gitignore
```

## Environment

The environment file `environment.yml` records the main experimental environment used during development, including Python 3.10, PyTorch 2.1.1, CUDA 11.8, and PyTorch Geometric 2.4.0.

If the CUDA version differs from the recorded environment, install the matching PyTorch and PyTorch Geometric packages for the target machine.

## Output Metrics

The implementation reports the final results averaged over the last several communication rounds:

- Client-Avg Test Acc
- Weighted Test Acc

Per-round accuracy and full-graph global-model accuracy are not printed by default in the repository version.

## Citation

```bibtex
@article{lu2026fedsha,
  title={FedSHA: Structure-aware Heterogeneity Adaptation for Personalized Federated Graph Node Classification},
  author={Lu, Ziyang},
  journal={Preprint},
  year={2026}
}
```

## License

The license will be updated according to the paper release plan.
