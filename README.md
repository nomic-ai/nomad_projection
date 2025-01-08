# NOMAD Projection
<img src="wiki_transparent.png" alt="NOMAD Projection" width="512">

Negative Or Mean Affinity Discrimination (NOMAD) Projection is a massively scalable method for nonlinear dimensionality reduction.
It is the fastest and easiest way to compute t-SNE or UMAP style visualizations of multi-million point datasets.

## Installation
You can install NOMAD Projection with pip:
```bash
pip install nomad-projection
```

## Usage
```python
from nomad_projection import NomadProjection

p = NomadProjection()

#Required Parameters
lowd = p.fit_transform(X=x,
                       epochs=100,
                       batch_size=80000)

#All Parameters
lowd = p.fit_transform(X=x,
                       epochs=100,
                       batch_size=80000,
                       n_neighbors=8,
                       n_noise=10000,
                       n_cells=5,
                       cluster_subset_size=5000000,
                       momentum=0.8,
                       lr_scale=0.1,
                       learning_rate_decay_start_time=0.3,
                       late_exaggeration_time=1.7,
                       late_exaggeration_scale=1.2,
                       late_exaggeration_n_noise=10000,
                       )
```

## Paper Replication

### Environment Setup
Due to the heterogeneous nature of python package management and cuda configuration, replicating the paper requires managing 3 different environments.

#### Nomad Projection Environment:
The nomad projection environment is the managed with venv.
Simply run the following commands from the root of the repository to create it:
```bash
python3 -m venv nomad_projection_env
source nomad_projection_env/bin/activate
pip install .
```

#### t-SNE-CUDA Environment:
The t-SNE-CUDA environment requires miniconda to be installed.
First, follow the instructions [here](https://docs.anaconda.com/miniconda/install/#quick-command-line-install) to install miniconda.
Then, follow the setup on the t-SNE-CUDA [repository](https://github.com/CannyLab/tsne-cuda).
Finally, run the following commands from the root of the repository in your conda environment:
```bash
conda install pytorch click scikit-learn pandas
pip install -e .
```

#### RAPIDS UMAP Environment:
The RAPIDS UMAP environment reuires a custom conda environment which is generated from the [RAPIDS installation selector](https://docs.rapids.ai/install/).
For the paper, the following command was used:
```bash
conda create -n rapids-24.10 -c rapidsai -c conda-forge -c nvidia  \
    rapids=24.10 python=3.12 'cuda-version>=12.0,<=12.5'
```
Once this command executes, run the following commands from the root of the repository in your rapisd conda environment:
```bash
conda install pytorch click scikit-learn pandas
pip install -e .
```

### Input Data
Input data can be downloaded from R2.
To gain access, run `aws configure` with the following credentials:

Access Key ID: `94bef7d178281190c5ca48f483b6504b`

Secret Access Key: `ac885a46694e8e1a073375b8da1961be42371a9f4434bd58ffd3e5c46a3be67b`

Then run the following command from the root of the repository to download the data (please note that this will download nearly a terabyte of data):
```bash
aws s3 sync --endpoint-url=https://9fa58365a1a3d032127970d0bd9a1290.r2.cloudflarestorage.com/ s3://nomad-projection-input-data ./data
```

### Figures
NOMAD Projection uses the figures submodule to manage generation of the figures in the paper.

#### ArXiv and Imagenet

Reproducing the arXiv and Imagenet figures requires two steps:
1. Run commands to generate results from each algorithm for each dataset.
2. Assemble the results into the final plot.


From the nomad projection environment, run the following command:
```python
python nomad_project/figures/arxiv.py --nomad
```
```python
python nomad_project/figures/imagenet.py --nomad
```

From the t-SNE-CUDA environment, run the following commands:
```python
python nomad_project/figures/arxiv.py --tsnecuda
```
```python
python nomad_project/figures/imagenet.py --tsnecuda
```

From the RAPIDS UMAP environment, run the following commands:
```python
python nomad_project/figures/arxiv.py --rapids-umap
```
```python
python nomad_project/figures/imagenet.py --rapids-umap
```

Finally, assemble the results into the final plot in the nomad projection environment
```python
python nomad_project/figures/arxiv.py --plot
```
```python
python nomad_project/figures/imagenet.py --plot
```
The output will be stored in the results directory in the root of the repository.

### PubMed
From the nomad projection environment, run the following command:
```python
python nomad_project/figures/pubmed.py --nomad
```
The output will be stored in the results directory in the root of the repository.

### Multilingual Wikipedia
```python
python nomad_project/figures/wiki.py --nomad
```
The output will be stored in the results directory in the root of the repository.