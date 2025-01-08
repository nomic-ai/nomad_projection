import os
import time
import glob
import click
import pickle
import numpy as np
from pathlib import Path
from nomad_projection.figures.eval import neighborhood_preservation_profile, random_triplet_accuracy, plot_metrics

def run_nomad(x):
    import torch
    from nomad_projection.projection import NomadProjection
    base_params = dict(X=x,
                       n_noise=10000,
                       n_neighbors=8,
                       late_exaggeration_time=1.7,
                       late_exaggeration_scale=1.2,
                       n_cells=5,
                       debug_plot=False)

    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print('Nomad Running on Multi GPU')
        configurations = {
                        'Nomad Multi GPU: 25 Epochs': dict(n_noise=10_000, epochs=25, lr_scale=0.1, batch_size=80_000),
                        'Nomad Multi GPU: 50 Epochs': dict(n_noise=10_000, epochs=50, lr_scale=0.1, batch_size=80_000),
                        'Nomad Multi GPU: 100 Epochs': dict(n_noise=10_000, epochs=100, lr_scale=0.1, batch_size=80_000),
                        'Nomad Multi GPU: 150 Epochs': dict(n_noise=10_000, epochs=150, lr_scale=0.1, batch_size=80_000),
                        }
    else:
        print('Nomad Running on Single GPU')
        configurations = {
                        'Nomad Single GPU: 10 Epochs': dict(n_noise=10_000, epochs=10, lr_scale=0.1, batch_size=90_000),
                        'Nomad Single GPU: 25 Epochs': dict(n_noise=10_000, epochs=25, lr_scale=0.1, batch_size=90_000),
                        'Nomad Single GPU: 50 Epochs': dict(n_noise=10_000, epochs=50, lr_scale=0.1, batch_size=90_000),
                        'Nomad Single GPU: 75 Epochs': dict(n_noise=10_000, epochs=75, lr_scale=0.1, batch_size=90_000),
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

        cur_metrics['time'] =  e_time - s_time
        cur_metrics['neighborhood_preservation'] = neighborhood_preservation_profile(x, low_d, k=15)
        cur_metrics['random_triplet_accuracy'] = random_triplet_accuracy(x, low_d)
        cur_metrics['projection'] = low_d
        metrics[config_name] = cur_metrics
        del p
    return metrics

def run_tsnecuda(x):
    from tsnecuda import TSNE
    metrics = {} 
    for n_iter in [5000, 20000, 35000, 50000]:

        cur_metrics = {}
        s = time.time()
        #following params set at https://github.com/CannyLab/tsne-cuda/blob/main/examples/cifar.py after poor default performance
        tsne = TSNE(n_components=2, verbose=True, n_iter=n_iter, perplexity=10000, num_neighbors=128)
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
    n_epochs_list = [50, 1000, 1750, 2500]
    for n_epochs in n_epochs_list:
        cur_metrics = {}
        s = time.time()
        umap = UMAP(n_components=2, n_epochs=n_epochs, n_neighbors=64, negative_sample_rate=100, verbose=True)
        low_d = umap.fit_transform(x)
        e = time.time()
        print('Time taken for RAPIDS UMAP {} epochs: {} seconds'.format(n_epochs, e - s))

        cur_metrics['neighborhood_preservation'] = neighborhood_preservation_profile(x, low_d, k=15)
        cur_metrics['random_triplet_accuracy'] = random_triplet_accuracy(x, low_d)
        cur_metrics['time'] = e - s
        cur_metrics['projection'] = low_d
        metrics['RAPIDS UMAP {} epochs'.format(n_epochs)] = cur_metrics
    return metrics

def embed_imagenet_with_clip():
    import torch
    from torchvision import datasets, transforms
    from torch.utils.data import DataLoader
    import open_clip
    from tqdm import tqdm
    from pathlib import Path
    import numpy as np

    # Load the OpenCLIP model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='laion2b_s34b_b79k', device=device)

    # Enable multi-GPU if available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)

    # Load the ImageNet dataset from HuggingFace
    from datasets import load_dataset
    dataset = load_dataset("imagenet-1k", split="train")
    
    # Create a proper dataset class with transform
    class ImageNetDataset(torch.utils.data.Dataset):
        def __init__(self, hf_dataset, transform):
            self.dataset = hf_dataset
            self.transform = transform
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            image = self.transform(item['image'])
            return image, item['label']
    
    dataset = ImageNetDataset(dataset, preprocess)
    dataloader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=4, pin_memory=True)

    # Prepare the directory to save embeddings
    save_dir = Path('nomad_project/data/imagenet_clip')
    save_dir.mkdir(parents=True, exist_ok=True)

    # Embed images and save embeddings along with their classes
    model.eval()
    try:
        for i, (images, classes) in enumerate(tqdm(dataloader, desc="Embedding Images")):
            images = images.to(device, non_blocking=True)
            with torch.no_grad():
                if isinstance(model, torch.nn.DataParallel):
                    embeddings = model.module.encode_image(images)
                else:
                    embeddings = model.encode_image(images)
                # Normalize embeddings
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                embeddings = embeddings.cpu().numpy()
            
            # Save embeddings and classes to file
            np.savez_compressed(save_dir / f'imagenet_embeddings_{i}.npz', 
                              embeddings=embeddings, 
                              classes=classes.numpy())
    except Exception as e:
        print(f"Error during embedding: {str(e)}")
        raise





@click.command()
@click.option('--nomad', is_flag=True, help='Run nomad algorithm')
@click.option('--tsnecuda', is_flag=True, help='Run t-SNE-CUDA algorithm')
@click.option('--rapids-umap', is_flag=True, help='Run RAPIDS UMAP algorithm')
@click.option('--preproc', is_flag=True, help='Preprocess ImageNet dataset with CLIP')
@click.option('--plot', is_flag=True, help='Plot metrics')
def main(nomad, tsnecuda, rapids_umap, preproc, plot):

    if preproc:
        embed_imagenet_with_clip()

    data_dir = Path(__file__).parent.parent.parent / 'data' / 'imagenet_clip'
    
    # Load all .npy files containing embeddings
    embedding_files = list(data_dir.glob('imagenet_embeddings_*.npz'))
    
    # Concatenate all embedding arrays
    embeddings_list = []
    for file in embedding_files:
        data = np.load(file)
        embeddings_list.append(data['embeddings'])
    x = np.concatenate(embeddings_list, axis=0)
    print(x.shape)

    # Prepare results directory
    results_dir = Path('./results/imagenet_clip')
    results_dir.mkdir(parents=True, exist_ok=True)

    if nomad:
        import torch
        num_gpus = torch.cuda.device_count()

        nomad_metrics = run_nomad(x)

        if num_gpus > 1:
            with open(results_dir / 'multi_gpu_nomad_metrics.pkl', 'wb') as f:
                pickle.dump(nomad_metrics, f)
        else:
            with open(results_dir / 'single_gpu_nomad_metrics.pkl', 'wb') as f:
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
        
        plot_metrics(metrics, results_dir, name='ImageNet')


if __name__ == '__main__':
    main()



