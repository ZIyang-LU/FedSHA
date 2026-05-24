# FedSHA

Official implementation of the paper "FedSHA: Structure-aware Heterogeneity Adaptation for Personalized Federated Graph Node Classification".

## Overview

FedSHA is a personalized federated graph learning method for node classification under both structural and statistical heterogeneity. The core idea is to perform structure-aware enhancement before personalized optimization, so that adaptive local aggregation is applied on an enhanced local graph view rather than on the original incomplete subgraph.


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

We evaluate all methods with 300 communication rounds, 5 local steps, 3 GNN layers, a Dirichlet concentration parameter of 0.1, and 2-hop structural enhancement when applicable. Experiments are conducted with client numbers in {5, 10, 15} and random seeds in {1971, 1972, 1999, 2025, 3407}. The learning rate is set to 0.1 for Cora and Citeseer, and 0.01 for Amazon-Computers.


## Environment

The environment file `environment.yml` records the main experimental environment used during development, including Python 3.10, PyTorch 2.1.1, CUDA 11.8, and PyTorch Geometric 2.4.0.

If the CUDA version differs from the recorded environment, install the matching PyTorch and PyTorch Geometric packages for the target machine.


Per-round accuracy and full-graph global-model accuracy are not printed by default in the repository version.


