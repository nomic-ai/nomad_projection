import os
import gc
import time
import torch
import numpy as np
from datetime import timedelta
import matplotlib.pyplot as plt
import torch.distributed as dist
import torch.multiprocessing as mp
from sklearn.decomposition import PCA
from matplotlib.animation import PillowWriter
from matplotlib.animation import FuncAnimation
from nomad_projection.neighbors import PartitionANN

def get_gpu_memory_usage():
    import gc
    total_memory = 0
    tensor_shapes = []
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                if obj.is_cuda:
                    memory = obj.element_size() * obj.nelement()
                    total_memory += memory
                    tensor_shapes.append((obj.shape, memory))
        except:
            pass
    
    print(f"GPU memory usage: {total_memory / 1024**2:.2f} MB")
    print("Tensor shapes:")
    for shape, mem in tensor_shapes:
        print(f"  Shape: {shape}, Memory: {mem / 1024**2:.2f} MB")


class NomadProjection:
    def __init__(self):
        self._knn = None
        self._model = None
        self._optim = None

        self.world_size = torch.cuda.device_count()
                    
        self.gpu_cluster_map = {}
        self._model_sizes = []
        self.cluster_assignments = None


    def _autocast_context(self):
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            props = torch.cuda.get_device_properties(device)
            if props.major >= 8:  # Ampere GPUs (compute capability >= 8.0) support BF16
                print(f"Using autocast for BF16 on GPU: {props.name}")
                return torch.autocast(device_type='cuda', dtype=torch.bfloat16)
            else:
                print(f"GPU {props.name} does not support BF16. Running without autocast.")
        else:
            print("CUDA is not available. Running without autocast.")

        return nullcontext() 


    @torch.compile
    def _step(self,
              model_idxs,
              knn,
              rank,
              batch_size,
              n_neighbors,
              n_noise,
              pos_weight,
              neg_weight,
              do_gather,
              context):

            x = torch.cat([self._model[model_num] for model_num in model_idxs], axis=0)
            mus = torch.stack([self._model[model_num].mean(dim=0) for model_num in model_idxs], axis=0)

            if do_gather:
                mu_container = []
                for i in range(self.world_size):
                    models_on_rank = len([ k for k, gpu in self.gpu_cluster_map.items() if gpu == i])
                    mu_container.append(torch.zeros(models_on_rank, mus.size(1), device=f'cuda:{rank}'))
                torch.distributed.all_gather(mu_container, mus)
                other_mus = torch.cat([m for i, m in enumerate(mu_container) if i != rank], dim=0)
                del mus

            n = x.size(0) 
            with context:
                target_idxs = torch.randint(low=0, high=n, size=(batch_size,), device=f'cuda:{rank}')
                noise_idxs = torch.randint(low=0, high=batch_size*n_neighbors, size=(batch_size, n_noise), device=f'cuda:{rank}')

                cur_batch_size = target_idxs.size(0)

                #Gather target and neighbor embeddings
                neighbor_idxs = knn[target_idxs, 1:n_neighbors+1]
                target_embs = x[target_idxs].reshape(cur_batch_size, 2)
                neighbor_embs = x[neighbor_idxs.reshape(cur_batch_size*n_neighbors, )]
                noise_embs = (neighbor_embs[noise_idxs]).reshape(cur_batch_size, n_noise, 2)
                
                #Compute kernel distances
                positives = ((target_embs.reshape(cur_batch_size, 1, 2) - neighbor_embs.reshape(cur_batch_size, n_neighbors, 2))**2).sum(axis=-1)
                negatives = ((target_embs.reshape(cur_batch_size, 1, 2) - noise_embs)**2).sum(axis=-1)

                poskerns = 1/(1+positives) #[Batch, Neighbors]
                negkerns_single = 1 / (1 + negatives)  # Shape: [B, L]
                negkerns = negkerns_single.sum(dim=1, keepdim=True).expand(-1, n_neighbors)  # Shape: [B, K]

                #Compute loss
                ranks = torch.arange(1, n_neighbors+ 1, dtype=torch.float32).cuda()
                exp_ranks = torch.exp(1 / ranks)
                sum_exp_ranks = exp_ranks.sum()
                pji = exp_ranks / sum_exp_ranks
                pji = pji.reshape(1, -1)

                if do_gather:
                    dist_negatives = ((target_embs.reshape(cur_batch_size, 1, 2) - other_mus)**2).sum(axis=-1)
                    dist_negkerns_single = 1 / (1 + dist_negatives)
                    dist_negkerns = dist_negkerns_single.sum(dim=1, keepdim=True).expand(-1, n_neighbors) 
                else:
                    dist_negkerns = torch.zeros_like(negkerns)

                losses = -1 * pos_weight * (torch.log(poskerns) * pji).sum(axis=-1) + neg_weight * (torch.log(poskerns + negkerns + dist_negkerns) * pji).sum(axis=-1)

                loss = losses.mean()
                
            loss.backward()

            self._optim.step()
            return loss.item()

    def train_on_gpu(self,
                     rank,
                     n,
                     batch_size,
                     epochs,
                     n_neighbors,
                     n_noise,
                     late_exaggeration_time,
                     late_exaggeration_scale,
                     late_exaggeration_n_noise,
                     lr_scale,
                     learning_rate_decay_start_time,
                     distributed):

        # derive schedules
        def n_noise_schedule(t):
            if t > late_exaggeration_time:
                return late_exaggeration_n_noise
            else:
                return n_noise

        def lr_schedule(t):
            if t > learning_rate_decay_start_time:
                tprime = (t-learning_rate_decay_start_time)/(1-learning_rate_decay_start_time)
                return n*lr_scale*(1-tprime) + n*1e-8*lr_scale*(tprime)
            else:
                return n*lr_scale

        def pos_weight_schedule(t):
            if t > late_exaggeration_time:
                return late_exaggeration_scale
            else:
                return 1

        if distributed:
            # setup distributed training
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '29500'

            torch.cuda.set_device(rank)
            dist.init_process_group(backend="nccl", rank=rank, world_size=self.world_size, timeout=timedelta(seconds=12000))

            print('Initialized Process Group: {}'.format(torch.cuda.current_device()))
            print(torch.cuda.get_device_name(torch.cuda.current_device()))

        context = self._autocast_context()

        model_idxs = [i for i in range(len(self._model)) if self.gpu_cluster_map[i] == rank]
        local_knn = []
        offset = 0
        for i in model_idxs:
            local_knn.append(torch.tensor(self._knn[i] + offset))
            offset += self._knn[i].shape[0]
        local_knn = torch.cat(local_knn, axis=0).to(f'cuda:{rank}')

        n_neighbors = torch.tensor(n_neighbors, device=f'cuda:{rank}')
        for epoch in range(epochs):
            
            t = epoch/epochs
            cur_n_noise = n_noise_schedule(t)
            cur_pos_weight = pos_weight_schedule(t)
            cur_lr = lr_schedule(t)
            for step in range(n//(batch_size * self.world_size)):
                self._optim.zero_grad()
                loss = self._step(model_idxs=model_idxs,
                                  knn=local_knn,
                                  rank=rank,
                                  batch_size=batch_size,
                                  n_neighbors=n_neighbors,
                                  n_noise=cur_n_noise,
                                  pos_weight=cur_pos_weight,
                                  do_gather=distributed,
                                  neg_weight=1,
                                  context=context)

                if not epoch % 2 and not rank:
                    print('t: {:.4f}'.format(t),
                          '\tdevice:{}'.format(rank),
                          '\tloss: {:.4f}'.format(loss),
                          '\tcur_lr: {}'.format(cur_lr))

                # Update learning rate and momentum
                for param_group in self._optim.param_groups:
                    param_group['lr'] = cur_lr


    def fit_transform(self,
            X,
            batch_size,
            epochs,
            n_cells=5,
            cluster_chunk_size=2000,
            n_neighbors=8,
            n_noise=10000,
            late_exaggeration_time=1.1,
            late_exaggeration_scale=1,
            late_exaggeration_n_noise=10000,
            momentum=0.8,
            learning_rate_decay_start_time=0.3,
            lr_scale=0.1,
            cluster_subset_size=5000000,
            debug_plot=False,
           ):

        #Setup Params
        n = X.shape[0]
        d = X.shape[1]

        self._knn_obj = PartitionANN(X, n_neighbors+1, n_cells, cluster_subset_size, cluster_chunk_size)
        self._knn = self._knn_obj._clusterwise_topk
        self._clusterwise_X_ids = self._knn_obj.clusterwise_X_ids
        self.cluster_assignments = self._knn_obj.labels
        torch.cuda.empty_cache()
            
        init_lr = n * lr_scale

        # Convert input data to a PyTorch tensor
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Perform PCA using PyTorch
        U, S, V = torch.pca_lowrank(X_tensor, q=2)
        init = U[:, :2].numpy()  # Get the first two principal components
        init /= (init[:, 0].std()) / 1e-4

        num_clusters = len(self._knn)
        cluster_ids = np.arange(num_clusters)
        
        # Initialize the embedding models on each GPU
        self._model = torch.nn.ParameterList()
        for cluster_id in cluster_ids:
            gpu = cluster_id % self.world_size
            init_idxs = self._clusterwise_X_ids[cluster_id]
            cluster_init_data = torch.tensor(init[init_idxs], dtype=torch.float32, device=f'cuda:{gpu}')
            cluster_model = torch.nn.Parameter(cluster_init_data)
            self._model_sizes.append(cluster_init_data.size(0))

            self._model.append(cluster_model)
            self.gpu_cluster_map[cluster_id] = gpu
        del self._clusterwise_X_ids

        self._optim = torch.optim.SGD([
            {'params': self._model.parameters()},
        ], lr=init_lr, momentum=momentum)

        distributed = self.world_size > 1
        if distributed:
            # Launch parallel training on all GPUs
            mp.spawn(self.train_on_gpu,
                    nprocs=self.world_size,
                    args=(n,
                        batch_size,
                        epochs,
                        n_neighbors,
                        n_noise,
                        late_exaggeration_time,
                        late_exaggeration_scale,
                        late_exaggeration_n_noise,
                        lr_scale,
                        learning_rate_decay_start_time,
                        distributed),
                    join=True)
        else:
            self.train_on_gpu(rank=0,
                              n=n,
                              batch_size=batch_size,
                              epochs=epochs,
                              n_neighbors=n_neighbors,
                              n_noise=n_noise,
                              late_exaggeration_time=late_exaggeration_time,
                              late_exaggeration_scale=late_exaggeration_scale,
                              late_exaggeration_n_noise=late_exaggeration_n_noise,
                              lr_scale=lr_scale,
                              learning_rate_decay_start_time=learning_rate_decay_start_time,
                              distributed=distributed)

        # Collect final embeddings in the order of input data
        final_embeddings = torch.zeros((len(self.cluster_assignments), 2), dtype=torch.float32)
        
        for cluster_id in range(len(self._model)):
            cluster_indices = torch.where(torch.tensor(self.cluster_assignments) == cluster_id)[0]
            cluster_embeddings = self._model[cluster_id].data.detach().cpu()
            final_embeddings[cluster_indices] = cluster_embeddings

        # Convert to numpy array before returning
        return final_embeddings.numpy()