import numpy as np
import torch
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
import pandas as pd

def get_eval_subsample(x, max_neighbors=20, subsample_size=5000, eval_idxs=None):
    # 1. Compute eval indexes
    if eval_idxs is None:
        eval_idxs = np.random.choice(x.shape[0], subsample_size, replace=False)
    
    # Convert data to PyTorch tensors and move to GPU
    eval_points = torch.tensor(x[eval_idxs], dtype=torch.float32).cuda()
    
    # 2. Compute nearest neighbors using CUDA
    chunk_size = 10000  # Adjust this based on available GPU memory
    top_k_distances, top_k_indices = torch.zeros(eval_points.shape[0], max_neighbors, device='cuda'), torch.zeros(eval_points.shape[0], max_neighbors, dtype=torch.long, device='cuda')
    top_k_distances.fill_(float('inf'))

    for i in range(0, x.shape[0], chunk_size):
        chunk = torch.tensor(x[i:i+chunk_size], dtype=torch.float32).cuda()
        chunk_distances = torch.cdist(eval_points, chunk)
        
        # Combine current chunk distances with existing top-k
        combined_distances = torch.cat([top_k_distances, chunk_distances], dim=1)
        combined_indices = torch.cat([top_k_indices, torch.arange(i, min(i+chunk_size, x.shape[0]), device='cuda').expand(eval_points.shape[0], -1)], dim=1)
        
        # Get top-k from combined results
        top_k_distances, top_k_idx = torch.topk(combined_distances, k=max_neighbors, dim=1, largest=False)
        top_k_indices = torch.gather(combined_indices, 1, top_k_idx)

    # Remove self-reference (first column)
    neighbor_idxs = top_k_indices[:, 1:max_neighbors]
    
    return {
        'eval_idxs': eval_idxs,
        'neighbor_idxs': neighbor_idxs.cpu().numpy(),
    }

def neighborhood_preservation(indices_X, indices_Y, k):
    indices_X = indices_X[:, :k]
    indices_Y = indices_Y[:, :k]

    preservation_scores = []
    for i in tqdm(range(indices_X.shape[0])):
        set_X = set(indices_X[i])
        set_Y = set(indices_Y[i])
        intersection_size = len(set_X.intersection(set_Y))
        preservation_score = intersection_size / k
        preservation_scores.append(preservation_score)

    
    return preservation_scores

def neighborhood_preservation_profile(x, y, k):
    x_eval_subsample = get_eval_subsample(x, max_neighbors=k+1)
    y_eval_subsample = get_eval_subsample(y, eval_idxs=x_eval_subsample['eval_idxs'], max_neighbors=k+1)
    
    scores = neighborhood_preservation(x_eval_subsample['neighbor_idxs'],
                                       y_eval_subsample['neighbor_idxs'],
                                       k)

    return {'scores': scores,
            'x_eval_subsample': x_eval_subsample,
            'y_eval_subsample': y_eval_subsample,
            'k': k,
            }

def plot_neighborhood_preservation_profiles(profiles, profile_names, save_dir, plotname):

    # Plot nps vs eval_ks for each number of datapoints
    plt.figure(figsize=(12, 6))
    for i, profile in enumerate(profiles):
        nps = profile['scores']
        eval_ks = profile['ks']
        means = [np.mean(e) for e in nps]
        ci_95 = [1.96 * np.std(e) / np.sqrt(len(e)) for e in nps]
        plt.errorbar(np.log2(eval_ks), means, yerr=ci_95, marker='o', capsize=5, label=profile_names[i])

    plt.xlabel('$log_{2}(K)$')
    plt.ylabel('Avg Neighborhood Preservation')
    plt.title(plotname)
    plt.legend()
    plt.grid(True)
    plt.savefig('{}/{}.png'.format(save_dir, plotname))

def random_triplet_accuracy(x, y, iters=5000):
    # Use random points from x and y instead of eval subsample
    num_points = min(len(x), len(y))  # Ensure we don't exceed the number of available points
    
    results = []    
    for _ in range(iters):
        # Randomly select three points
        a, b, c = np.random.choice(num_points, 3, replace=False)
        
        # Calculate distances in original space
        dist_ab_x = np.linalg.norm(x[a] - x[b])
        dist_ac_x = np.linalg.norm(x[a] - x[c])
        
        # Calculate distances in projected space
        dist_ab_y = np.linalg.norm(y[a] - y[b])
        dist_ac_y = np.linalg.norm(y[a] - y[c])
        
        # Check if relative distances are preserved
        if (dist_ab_x < dist_ac_x and dist_ab_y < dist_ac_y) or (dist_ab_x > dist_ac_x and dist_ab_y > dist_ac_y):
            results.append(1)
        else:
            results.append(0)
    
    return results


def plot_metrics(metrics, save_dir, name, k=15):
    # Set larger font sizes for all plots
    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'axes.titlesize': 20,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
    })


    # Plot random triplet accuracy vs time
    plt.figure(figsize=(10, 6))
    
    for algo_name, algo_metrics in metrics.items():
        if 'rapids' in algo_name:
            algo_name = 'RapidsUMAP 1xH100'
            line_style = '-'
            line_color = 'red'
        elif 'nomic' in algo_name:
            if 'multi' in algo_name:
                algo_name = 'NOMAD 2xH100'
                line_style = '--'
                line_color = 'green'
            else:
                algo_name = 'NOMAD 1xH100'
                line_style = '-'
                line_color = 'green'
        elif 'tsnecuda' in algo_name:
            algo_name = 't-SNE-CUDA 1xH100'
            line_style = '-'
            line_color = 'blue'

        cur_times = []
        cur_accs = []
        cur_cis = []
        for hparams, data in algo_metrics.items():
            cur_times.append(data['time'])
            cur_accs.append(np.mean(data['random_triplet_accuracy']))
            cur_ci = 1.96 * np.std(data['random_triplet_accuracy']) / np.sqrt(len(data['random_triplet_accuracy']))
            cur_cis.append(cur_ci)

        plt.errorbar(cur_times, cur_accs, yerr=cur_cis, marker='o', capsize=5, label=algo_name, linestyle=line_style, color=line_color)

    plt.xlabel('Time (seconds)')
    plt.ylabel('Random Triplet Accuracy')
    plt.title(f'{name}: Random Triplet Accuracy vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{save_dir}/random_triplet_accuracy_vs_time.png')
    plt.close()

    # Plot neighborhood preservation vs time
    plt.figure(figsize=(10, 6))
    
    for algo_name, algo_metrics in metrics.items():
        if 'rapids' in algo_name:
            algo_name = 'RapidsUMAP 1xH100'
            line_style = '-'
            line_color = 'red'
        elif 'nomic' in algo_name:
            if 'multi' in algo_name:
                algo_name = 'NOMAD 2xH100'
                line_style = '--'
                line_color = 'green'
            else:
                algo_name = 'NOMAD 1xH100'
                line_style = '-'
                line_color = 'green'
        elif 'tsnecuda' in algo_name:
            algo_name = 't-SNE-CUDA 1xH100'
            line_style = '-'
            line_color = 'blue'


        cur_times = []
        cur_accs = []
        cur_cis = []
        for hparams, data in algo_metrics.items():
            cur_times.append(data['time'])
            np_scores = data['neighborhood_preservation']['scores']
            np_mean = np.mean(np_scores)
            cur_accs.append(np_mean)

            np_std = np.std(np_scores)
            np_ci = 1.96 * np_std / np.sqrt(len(np_scores))  # 95% confidence interval
            cur_cis.append(np_ci)
            
        plt.errorbar(cur_times, cur_accs, yerr=cur_cis, marker='o', capsize=5, label=algo_name, linestyle=line_style, color=line_color)
    
    plt.xlabel('Time (seconds)')
    plt.ylabel(f'Neighborhood Preservation @ {k}')
    plt.title(f'{name}: Neighborhood Preservation vs Time')
    plt.legend()
    plt.grid(True)
    
    plt.savefig(f'{save_dir}/neighborhood_preservation_vs_time.png')
    plt.close()
    
