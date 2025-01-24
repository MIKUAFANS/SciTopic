import os

import numpy as np
from sklearn.cluster import KMeans, HDBSCAN


default_n_threads = 8
os.environ['OPENBLAS_NUM_THREADS'] = f"{default_n_threads}"
os.environ['MKL_NUM_THREADS'] = f"{default_n_threads}"
os.environ['OMP_NUM_THREADS'] = f"{default_n_threads}"

class TextCluster:
    def __init__(self, model_name="k_means", n_clusters : int = None, **kwargs):
        self.model_name = model_name
        self.n_clusters = n_clusters

        if model_name == "k_means":
            assert n_clusters is not None, "n_clusters must be provided"
            self.model = KMeans(n_clusters=n_clusters, random_state=2024)
        elif model_name == "hdbscan":
            self.model = HDBSCAN(**kwargs)
        else:
            raise ValueError("Invalid model name")
        
    def fit(self, embeddings):
        self.model.fit(embeddings)
        return self.model.labels_
    
    def get_cluster_centers(self):
        if self.model_name == "hdbscan":
            return None
        return self.model.cluster_centers_
    
    def save(self, save_path : str):
        if save_path is None:
            save_path = os.path.join(os.getcwd(), "output", "cluster")
        
        os.makedirs(save_path, exist_ok=True)
        if self.model_name == "hdbscan":
            np.save(f'{save_path}/cluster_labels_hdbscan_{self.n_clusters}.npy', self.model.labels_)
        else:
            np.save(f'{save_path}/cluster_centers_k_means_{self.n_clusters}.npy', self.model.cluster_centers_)
            np.save(f'{save_path}/cluster_labels_{self.n_clusters}.npy', self.model.labels_)

    def __call__(self, embeddings, save_path : str = None):
        self.fit(embeddings)
        self.save(save_path)
        return self.model.labels_, self.get_cluster_centers()
    

