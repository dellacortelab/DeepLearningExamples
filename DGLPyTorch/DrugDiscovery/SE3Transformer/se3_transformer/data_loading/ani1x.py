import dgl
import pathlib
import torch
from dgl import DGLGraph
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm
import h5py
import os
import numpy as np

from se3_transformer.data_loading.data_module import DataModule
from se3_transformer.model.basis import get_basis
from se3_transformer.runtime.utils import get_local_rank, str2bool, using_tensor_cores


class ANI1xDataModule(DataModule):
    NODE_FEATURE_DIM = 4
    EDGE_FEATURE_DIM = 0

    def __init__(self,
                 data_dir: pathlib.Path,
                 batch_size: int = 256,
                 num_workers: int = 8,
                 num_degrees: int = 4,
                 amp: bool = False,
                 precompute_bases: bool = False,
                 knn: int = None,
                 **kwargs):
        self.data_dir = data_dir # This needs to be before __init__ so that prepare_data has access to it
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.amp = amp
        self.batch_size = batch_size
        self.num_degrees = num_degrees 

        self.ds_train = self.get_dataset(mode='train', knn=knn)
        self.ds_val = self.get_dataset(mode='validation', knn=knn)
        self.ds_test = self.get_dataset(mode='test', knn=knn)

        self.energy_mean = 0.0184
        self.energy_std = 0.1062
        self.force_std = 0.0709

    def prepare_data(self):
        species_list = []
        pos_list = []
        forces_list = []
        energy_list = []
        num_list = []

        file_path = os.path.join(self.data_dir, 'ani1xrelease.h5')
        it = self.iter_data_buckets(file_path, keys=['wb97x_dz.forces', 'wb97x_dz.energy'])
        for num, molecule in enumerate(it):
            species = molecule['atomic_numbers']
            for pos, energy, forces in zip(molecule['coordinates'], molecule['wb97x_dz.energy'], molecule['wb97x_dz.forces']):
                pos_list.append(pos)
                species_list.append(species)
                energy_list.append(energy)
                forces_list.append(forces)
                num_list.append(num)
        
        self.species_list = species_list
        self.pos_list = pos_list
        self.forces_list = forces_list
        self.energy_list = energy_list
        self.num_list = num_list

    @staticmethod 
    def iter_data_buckets(h5filename, keys=['wb97x_dz.energy']):
        """ Iterate over buckets of data in ANI HDF5 file. 
        Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
        and other available properties specified by `keys` list, w/o NaN values.
        """
        keys = set(keys)
        keys.discard('atomic_numbers')
        keys.discard('coordinates')
        with h5py.File(h5filename, 'r') as f:
            for grp in f.values():
                Nc = grp['coordinates'].shape[0]
                mask = np.ones(Nc, dtype=np.bool)
                data = dict((k, grp[k][()]) for k in keys)
                for k in keys:
                    v = data[k].reshape(Nc, -1)
                    mask = mask & ~np.isnan(v).any(axis=1)
                if not np.sum(mask):
                    continue
                d = dict((k, data[k][mask]) for k in keys)
                d['atomic_numbers'] = grp['atomic_numbers'][()]
                d['coordinates'] = grp['coordinates'][()][mask]
                yield d

    def get_dataset(self, mode='train', knn=None):
        if mode=='train':
            idx = np.arange(18)
        elif mode=='validation':
            idx = np.array([18])
        elif mode=='test':
            idx = np.array([19])

        ds_pos_list = []
        ds_species_list = []
        ds_energy_list = []
        ds_forces_list = []

        for pos, species, energy, forces, num in zip(self.pos_list, self.species_list, self.energy_list, self.forces_list, self.num_list):
            if num%20 in idx:
                ds_pos_list.append(pos)
                ds_species_list.append(species)
                ds_energy_list.append(energy) 
                ds_forces_list.append(forces)

        dataset = ANI1xDataset(ds_pos_list, ds_species_list, ds_energy_list, ds_forces_list, knn=None)
        return dataset

    def _collate(self, samples):
        graphs, node_feats_list, targets_list, *bases = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)

        # node features
        species = torch.cat([node_feats['0'] for node_feats in node_feats_list])
        node_feats = {'0': species}

        # edge feats
        edge_feats = {}

        # targets
        energy = torch.cat([targets['0'] for targets in targets_list])
        forces = torch.cat([targets['1'] for targets in targets_list])
        targets = {'0': energy,
                   '1': forces}

        return batched_graph, node_feats, edge_feats, targets

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ANI1x dataset")
        parser.add_argument('--precompute_bases', type=str2bool, nargs='?', const=True, default=False,
                            help='Precompute bases at the beginning of the script during dataset initialization,'
                                 ' instead of computing them at the beginning of each forward pass.')
        parser.add_argument('--num_channels', type=int, default=32,
                            help='Number of hidden channels in model')
        parser.add_argument('--num_degrees', type=int, default=3,
                            help='Number of degrees in model')
        return parent_parser

    def __repr__(self):
        return 'ANI1x'

    @staticmethod
    def get_bound_idx(energy_targets):
        bound_idx = torch.where(energy_targets[:-1] != energy_targets[1:])[0]
        last = torch.tensor([len(energy_targets), device=bound_idx.device])
        bound_idx = torch.cat([bound_idx, last])
        return bound_idx

    @staticmethod
    def loss_fn(pred, target):
        bound_idx = ANI1xDataModule.get_bound_idx(target['0'])
        energy_loss = ANI1xDataModule.energy_loss_fn(pred['0'], target['0'], bound_idx)
        force_loss = ANI1xDataModule.force_loss_fn(pred['1'], target['1'], bound_idx)
        loss = energy_loss + force_loss
        return loss

    @staticmethod
    def energy_loss_fn(pred, target, bound_idx):
        start = 0
        energy_losses = torch.zeros(len(bound_idx), device=pred.device)
        for i, stop in enumerate(bound_idx):
            energy_losses[i] = F.mse_loss(torch.sum(pred[start:stop]), target[start])
            start = stop
        return torch.mean(energy_losses)

    @staticmethod
    def force_loss_fn(pred, target, bound_idx):
        start = 0
        force_losses = torch.zeros(len(bound_idx), device=pred.device)
        for i, stop in enumerate(bound_idx):
            force_losses[i] = F.mse_loss(pred[start:stop], target[start:stop])
            start = stop
        return torch.mean(force_losses)

    @staticmethod
    def energy_mae_fn(pred, target, bound_idx):
        start = 0 
        energy_maes = torch.zeros(len(bound_idx), device=pred.device)
        for i, l in enumerate(bound_idx):
            energy_maes[i] = torch.abs(torch.sum(pred[start:stop]) - target[start])
            start = stop
        return torch.mean(force_maes)

    @staticmethod
    def force_mae_fn(pred, target, bound_idx):
        start = 0 
        force_maes = torch.zeros(len(bound_idx), device=pred.device)
        for i, l in enumerate(bound_idx):
            force_maes[i] = torch.mean(torch.abs(pred[start:stop] - target[start:stop]))
            start = stop
        return torch.mean(force_maes)



class ANI1xDataset(Dataset):
    def __init__(self, pos_list, species_list, energy_list, forces_list, knn=None, normalize=True):
        self.pos_list = pos_list
        self.species_list = species_list
        self.energy_list = energy_list
        self.forces_list = forces_list
        self.knn = knn
        self.normalize = normalize

        eye = torch.eye(4)
        self.species_dict = {1: eye[0], 6: eye[1], 7: eye[2], 8: eye[3]}

        self.energy_mean = 0.0184
        self.energy_std = 0.1062
        self.force_std = 0.0709

    def __len__(self):
        return len(self.pos_list)

    def __getitem__(self, i):
        pos = self.pos_list[i]
        species = self.species_list[i]
        forces = self.forces_list[i]
        energy = self.energy_list[i]

        # Create graph
        pos = torch.tensor(pos)
        graph = self._create_graph(pos, knn=self.knn)

        # Create node features
        species = torch.stack([self.species_dict[atom] for atom in species])
        node_feats = {'0': species.unsqueeze(-1)}

        # Create targets
        if self.normalize:
            energy = (energy - self.energy_mean)/self.energy_std
            forces = forces/self.force_std
        energy = torch.ones(len(pos)) * energy
        forces = torch.tensor(forces)
        targets = {'0': energy,
                   '1': forces}

        return graph, node_feats, targets

    @staticmethod 
    def _create_graph(pos, knn=None):
        u = []
        v = []
        
        if knn is None or len(pos)<knn:
            nbrs = NearestNeighbors(n_neighbors=len(pos)).fit(pos)
        else:
            nbrs = NearestNeighbors(n_neighbors=knn).fit(pos)

        distances, indices = nbrs.kneighbors(pos)
        for idx_list in indices:
            for k in idx_list[1:]:
                v.append(idx_list[0])
                u.append(k)

        graph = dgl.graph((u,v))
        graph.ndata['pos'] = pos
        graph.edata['rel_pos'] = pos[v] - pos[u]
        return graph 
