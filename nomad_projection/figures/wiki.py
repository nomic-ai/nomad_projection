import glob 
import pickle
import time
import click
import numpy as np
import pandas as pd
from torch.nn.parallel import DataParallel

import torch
import multiprocessing
from tqdm import tqdm
from pathlib import Path

from nomad_projection.figures.eval import neighborhood_preservation_profile, random_triplet_accuracy, plot_metrics


def preproc():
    from datasets import load_dataset, get_dataset_config_names
    from sentence_transformers import SentenceTransformer

    # Set the start method to 'spawn'
    multiprocessing.set_start_method('spawn', force=True)

    # Check if CUDA is available and set the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize the model
    model = SentenceTransformer("BAAI/bge-m3")

    # Use DataParallel if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = DataParallel(model)

    model.to(device)  # Move the model to GPU(s)


    def embed_batch(batch):
        # Ensure the input is on the correct device
        return model.module.encode(batch['text'], show_progress_bar=False) if isinstance(model, DataParallel) else model.encode(batch['text'], show_progress_bar=False)

    data_dir = Path(__file__).parent.parent.parent / 'data' / 'wiki'
    data_dir.mkdir(parents=True, exist_ok=True)

    subsets = get_dataset_config_names("wikimedia/wikipedia")

    def process_subset(subset, gpu_id, done_queue):
        # Set the device for this process
        device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
        print(f"Processing {subset} on {device}")

        # Initialize the model for this process
        model = SentenceTransformer("BAAI/bge-m3")
        model.to(device)

        def embed_batch(batch):
            return model.encode(batch['text'], show_progress_bar=False)

        # Embed the dataset in batches and save in shards
        shard_files = list(data_dir.glob(f'{subset}_embeddings_shard_*.npy'))
        if not shard_files:
            batch_size = 100000
            shard_size = 1000000  
            shard_count = 0
            current_shard = []

            ds = load_dataset("wikimedia/wikipedia", subset)
            for i in tqdm(range(0, len(ds['train']), batch_size), desc=f"Processing {subset}"):
                batch = ds['train'][i:i+batch_size]
                batch_embeddings = embed_batch(batch)
                current_shard.extend(batch_embeddings)
                
                if len(current_shard) >= shard_size:
                    # Save the current shard
                    shard_file = data_dir / f'{subset}_embeddings_shard_{shard_count}.npy'
                    np.save(shard_file, np.array(current_shard))
                    print(f"Saved shard {shard_count} to {shard_file}")
                    
                    # Reset for next shard
                    current_shard = []
                    shard_count += 1

            # Save any remaining embeddings in the last shard
            if current_shard:
                shard_file = data_dir / f'{subset}_embeddings_shard_{shard_count}.npy'
                np.save(shard_file, np.array(current_shard))
                print(f"Saved final shard {shard_count} to {shard_file}")

            print(f"Processed and saved embeddings for subset: {subset} in {shard_count + 1} shards")
            done_queue.put(gpu_id)  # Signal that this GPU is now free

    def launch_preproc():
        # Get the number of available GPUs
        num_gpus = torch.cuda.device_count()
        print(f"Number of available GPUs: {num_gpus}")

        subsets = get_dataset_config_names("wikimedia/wikipedia")
        done_queue = multiprocessing.Queue()
        processes = []
        active_gpus = set()

        for subset in subsets:
            # Wait for a free GPU
            while len(active_gpus) >= num_gpus:
                free_gpu = done_queue.get()
                active_gpus.remove(free_gpu)

            # Find the first available GPU
            for gpu_id in range(num_gpus):
                if gpu_id not in active_gpus:
                    break
            
            active_gpus.add(gpu_id)
            p = multiprocessing.Process(target=process_subset, args=(subset, gpu_id, done_queue))
            p.start()
            processes.append(p)

        # Wait for all processes to complete
        for _ in processes:
            done_queue.get()

        # Ensure all processes have finished
        for p in processes:
            p.join()

        print("All subsets processed.")

def run_nomad(x):
    from nomad_project.projection import NomadProjection
    base_params = dict(X=x,
                       n_noise=10000,
                       n_neighbors=8,
                       late_exaggeration_time=1.7,
                       momentum=0.0,
                       n_cells=128,
                       batch_size=50_000,
                       epochs=400,
                       debug_plot=False)

    configurations = {
                      'Nomad 600 Epochs': dict(n_noise=10_000,
                                               n_neighbors=64,
                                               epochs=600,
                                               lr_scale=0.15,
                                               learning_rate_decay_start_time=0.1,
                                               late_exaggeration_time=0.7,
                                               late_exaggeration_scale=1.5,
                                               batch_size=70_000,
                                               cluster_subset_size=2500000,
                                               cluster_chunk_size=1000),
                      }
    metrics = {}

    for config_name, configuration in configurations.items():
        cur_metrics = {}

        current_kwargs = {**base_params, **configuration}
        p = NomadProject()
        s_time = time.time()
        low_d = p.fit_transform(**current_kwargs)
        e_time = time.time()
        print(f'Time taken for {config_name}: {e_time - s_time:.2f} seconds')

        cur_metrics['time'] =  e_time - s_time
        #TODO is 10 so we can compare to dmitry
        cur_metrics['neighborhood_preservation'] = neighborhood_preservation_profile(x, low_d, k=10)
        cur_metrics['random_triplet_accuracy'] = random_triplet_accuracy(x, low_d)
        cur_metrics['projection'] = low_d
        metrics[config_name] = cur_metrics
        del p
    return metrics

def upload_results(metrics, results_dir):

    print(metrics.keys())
    subsets = sorted(get_dataset_config_names("wikimedia/wikipedia"))
    lowd = metrics['Nomad 500 Epochs']['projection']

    df_list = []
    for i, subset in tqdm(enumerate(subsets), desc="Loading subsets"):
        ds = load_dataset("wikimedia/wikipedia", subset)
        df_list.append(pd.DataFrame({
            'title': ds['train']['title'],
            'subset': subset,
            'url': ds['train']['url'],
            'wid': ds['train']['id']
        }))

    meta = pd.concat(df_list, ignore_index=True)
    print(meta.head())  # Display the first few rows of the dataframe

    from nomic import atlas
    atlas.map_data(embeddings=lowd,
                   data=meta,
                   topic_model=False,
                   identifier='brandon/wikifulltest',
                   duplicate_detection=False)





@click.command()
@click.option('--preproc', is_flag=True, help='Run the preproc')
@click.option('--debug', is_flag=True, help='Run the preproc in debug mode')
@click.option('--nomad', is_flag=True, help='Run nomad algorithm')
@click.option('--plot', is_flag=True, help='Plot metrics')
@click.option('--upload', is_flag=True, help='Upload results')
def main(preproc, debug, nomad, plot, upload):

    if preproc:
        launch_preproc()


    results_dir = Path('./results/wiki')
    # Prepare results directory
    results_dir.mkdir(parents=True, exist_ok=True)

    if nomad:
    
        # Load and stack all .npy files from the preproc output directory
        data_dir = Path(__file__).parent.parent.parent / 'data' / 'wiki'
        npy_files = sorted(data_dir.glob('*.npy'))
        
        if not npy_files:
            raise FileNotFoundError("No .npy files found in the data directory.")
        
        # Check if "x.npy" exists in the data directory
        x_file = data_dir / 'x.npy'
        if debug:
            import numba as nb
            import random
            @nb.njit('float64[:,:](int_, int_)', parallel=True)
            def genRandom(n, m):
                res = np.empty((n, m))

                # Parallel loop
                for i in nb.prange(n):
                    for j in range(m):
                        res[i, j] = np.random.rand()

                return res
            s = time.time()
            x = genRandom(60000000, 1024)
            e = time.time()
        elif x_file.exists():
            print("Loading pre-stacked data ")
            s = time.time()
            x = np.load(x_file)
            e = time.time()
            print(f'Loading took: {e - s:.2f} seconds')
        else:
            print("x.npy not found. Loading and combining individual embedding files.")
            # Preallocate the array
            embedding_files = [f for f in npy_files if 'embeddings' in f.name]
            # Sort the files alphabetically
            embedding_files.sort()

            x = np.empty((61614907, 1024))
            start_idx = 0
            for npy_file in tqdm(npy_files):
                current_data = np.load(npy_file)
                end_idx = start_idx + current_data.shape[0]
                x[start_idx:end_idx] = current_data
                start_idx = end_idx

            np.save(x_file, x)
            print(f"Saved combined embeddings to {x_file}")
        
        print(f"Final loaded data shape: {x.shape}")

        nomad_metrics = run_nomad(x)
        with open(results_dir / 'nomad_metrics.pkl', 'wb') as f:
            pickle.dump(nomad_metrics, f)


    if plot:
        # Load all results using glob
        metrics = {} 
        
        # Find all metric pickle files
        metric_files = glob.glob(str(results_dir / '*_metrics.pkl'))

        # Load and merge all found metric files
        for file in metric_files:
            with open(file, 'rb') as f:
                metrics[file] = pickle.load(f)
        
        plot_metrics(metrics, results_dir, name='Wiki')

    if upload:
        metrics_file = results_dir / 'nomad_metrics.pkl'
        if metrics_file.exists():
            print("Loading existing nomad_metrics from results directory.")
            with open(metrics_file, 'rb') as f:
                nomad_metrics = pickle.load(f)
            upload_results(nomad_metrics, results_dir)
        else:
            print("nomad_metrics.pkl not found in results directory.")

if __name__ == '__main__':
    main()
