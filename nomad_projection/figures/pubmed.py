import os
import gc
import time
import glob
import torch
import click
import pickle
import numpy as np
from pathlib import Path
from nomad_projection.figures.eval import neighborhood_preservation_profile, random_triplet_accuracy, plot_metrics

def run_nomad(x):
    from nomad_projection.projection import NomadProjection
    base_params = dict(X=x,
                       n_noise=10000,
                       n_neighbors=8,
                       late_exaggeration_time=1.7,
                       momentum=0.0,
                       n_cells=16,
                       batch_size=50_000,
                       epochs=400,
                       debug_plot=False)

    configurations = {
                      'Nomad 600 Epochs': dict(n_noise=10_000,
                                               n_neighbors=8,
                                               n_cells=16,
                                               epochs=600,
                                               lr_scale=0.2,
                                               batch_size=80_000,
                                               cluster_subset_size=2500000,
                                               cluster_chunk_size=1000),
                      }

    metrics = {}

    for config_name, configuration in configurations.items():
        cur_metrics = {}

        current_kwargs = {**base_params, **configuration}
        p = NomadProjection()
        s_time = time.time()
        low_d = p.fit_transform(**current_kwargs)
        e_time = time.time()
        print(f'Time taken for {config_name}: {e_time - s_time:.2f} seconds')
        # Free GPU memory
        torch.cuda.empty_cache()
        for obj in gc.get_objects():
            if torch.is_tensor(obj):
                if obj.is_cuda:
                    del obj
        # Print out CUDA memory usage
        print(f"CUDA memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"CUDA memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        cur_metrics['time'] =  e_time - s_time
        #NOTE 10 so we can compare to dmitry
        cur_metrics['neighborhood_preservation'] = neighborhood_preservation_profile(x, low_d, k=10)
        cur_metrics['random_triplet_accuracy'] = random_triplet_accuracy(x, low_d)
        cur_metrics['projection'] = low_d
        metrics[config_name] = cur_metrics
        del p
    return metrics

def run_tsnecuda(x):
    from tsnecuda import TSNE
    metrics = {} 
    for n_iter in [75000, 125000, 175000]:

        cur_metrics = {}
        s = time.time()
        tsne = TSNE(n_components=2, verbose=True, n_iter=n_iter)
        low_d = tsne.fit_transform(x)
        e = time.time()
        cur_metrics['neighborhood_preservation'] = neighborhood_preservation_profile(x, low_d, k=15)
        cur_metrics['random_triplet_accuracy'] = random_triplet_accuracy(x, low_d)
        cur_metrics['time'] = e - s
        cur_metrics['projection'] = low_d
        metrics['t-SNE-CUDA {} iter'.format(n_iter)] = cur_metrics
    return metrics


def run_rapids_umap(x):
    import cuml
    from cuml.manifold import UMAP

    metrics = {}
    n_epochs_list = [50, 100, 150]
    for n_epochs in n_epochs_list:
        cur_metrics = {}
        s = time.time()
        umap = UMAP(n_components=2, n_epochs=n_epochs)
        low_d = umap.fit_transform(x)
        e = time.time()

        cur_metrics['neighborhood_preservation'] = neighborhood_preservation_profile(x, low_d, k=15)
        cur_metrics['random_triplet_accuracy'] = random_triplet_accuracy(x, low_d)
        cur_metrics['time'] = e - s
        cur_metrics['projection'] = low_d
        metrics['RAPIDS UMAP {} epochs'.format(n_epochs)] = cur_metrics
    return metrics


@click.command()
@click.option('--nomad', is_flag=True, help='Run nomad algorithm')
@click.option('--tsnecuda', is_flag=True, help='Run t-SNE-CUDA algorithm')
@click.option('--rapids-umap', is_flag=True, help='Run RAPIDS UMAP algorithm')
@click.option('--plot', is_flag=True, help='Plot metrics')
def main(nomad, tsnecuda, rapids_umap, plot):

    
    # Check if data directory exists, create if not
    data_dir = Path(__file__).parent.parent.parent / 'data' / 'pubmed'
    data_dir.mkdir(parents=True, exist_ok=True)

    # Path for the numpy file
    data_file = data_dir / 'PubMedBERT_embeddings_float16_2024.npy'

    print(f"Loading existing data from {data_file}")

    # Load the data
    x = np.load(data_file)

    # Prepare results directory
    results_dir = Path('./results/pubmed')
    results_dir.mkdir(parents=True, exist_ok=True)

    if nomad:
        nomad_metrics = run_nomad(x)
        with open(results_dir / 'nomad_metrics.pkl', 'wb') as f:
            pickle.dump(nomad_metrics, f)

    if tsnecuda:
        tsnecuda_metrics = run_tsnecuda(x)
        with open(results_dir / 'tsnecuda_metrics.pkl', 'wb') as f:
            pickle.dump(tsnecuda_metrics, f)

    if rapids_umap:
        rapids_umap_metrics = run_rapids_umap(x)
        with open(results_dir / 'rapids_umap_metrics.pkl', 'wb') as f:
            pickle.dump(rapids_umap_metrics, f)


    if plot:
        # Load all results using glob
        metrics = {} 
        
        # Find all metric pickle files
        metric_files = glob.glob(str(results_dir / '*_metrics.pkl'))

        # Load and merge all found metric files
        for file in metric_files:
            with open(file, 'rb') as f:
                metrics[file] = pickle.load(f)
        
        plot_metrics(metrics, results_dir, name='PubMed', k=10)


if __name__ == '__main__':
    main()
