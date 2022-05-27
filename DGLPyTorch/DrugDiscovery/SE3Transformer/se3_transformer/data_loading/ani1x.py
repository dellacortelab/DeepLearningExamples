import dgl
import pathlib
import torch
from dgl import DGLGraph
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import random_split, DataLoader, Dataset
from tqdm import tqdm
import h5py
import os
import numpy as np
from collections import Counter

from se3_transformer.data_loading.data_module import DataModule
from se3_transformer.model.basis import get_basis
from se3_transformer.runtime.utils import get_local_rank, str2bool, using_tensor_cores

ATOM_ENERGIES = torch.tensor([
                             -0.600952980000,
                             -38.08316124000,
                             -54.70775770000,
                             -75.19446356000])

ENERGY_STD = 0.1062

ENERGY_STD = 1.643134
NUM_SPECIES = 4



class ANI1xDataModule(DataModule):
    NUM_SPECIES = NUM_SPECIES
    ENERGY_STD = ENERGY_STD
    def __init__(self,
                 data_dir: pathlib.Path,
                 batch_size: int = 256,
                 num_workers: int = 8,
                 num_degrees: int = 4,
                 amp: bool = False,
                 precompute_bases: bool = False,
                 cutoff: float = float('inf'),
                 **kwargs):
        self.data_dir = data_dir # This needs to be before __init__ so that prepare_data has access to it
        super().__init__(batch_size=batch_size, num_workers=num_workers, collate_fn=self._collate)
        self.amp = amp
        self.batch_size = batch_size
        self.num_degrees = num_degrees 

        self.load_main_data()
        self.load_test_data()
        self.ds_train = self.get_dataset(mode='train', cutoff=cutoff)
        self.ds_val = self.get_dataset(mode='validation', cutoff=cutoff)
        self.ds_test = self.get_dataset(mode='test', cutoff=cutoff)

    def load_main_data(self):
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
        return

    def load_test_data(self):
        species_list = []
        pos_list = []
        forces_list = []
        energy_list = []
        num_list = []

        test_dir = os.path.join(self.data_dir, 'COMP6v1')
        for subdir in os.listdir(test_dir):
            subpath = os.path.join(test_dir, subdir)
            for file_name in os.listdir(subpath):
                filepath = os.path.join(subpath, file_name)
                with h5py.File(filepath, 'r') as f:
                    for main in f.values():
                        for mol in main.values():
                            species_list += [list(mol['species'])] * len(mol['forces'])
                            pos_list += list(mol['coordinates'])
                            forces_list += list(mol['forces'])
                            energy_list += list(mol['energies'])

        self.test_species_list = species_list
        self.test_pos_list = pos_list
        self.test_forces_list = forces_list
        self.test_energy_list = energy_list
        return

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

    def get_dataset(self, mode='train', cutoff=float('inf')):
        if mode=='test':
            dataset = ANI1xDataset(self.test_pos_list,
                                   self.test_species_list,
                                   self.test_energy_list,
                                   self.test_forces_list,
                                   cutoff=cutoff,
                                   test=True
            )
            return dataset

        elif mode=='train':
            idx = np.arange(19)
        elif mode=='validation':
            idx = np.array([19])

        ds_pos_list = []
        ds_species_list = []
        ds_energy_list = []
        ds_forces_list = []

        for pos, species, energy, forces, num in zip(self.pos_list,
                                                     self.species_list,
                                                     self.energy_list,
                                                     self.forces_list,
                                                     self.num_list):
            if num % 20 in idx:
                ds_pos_list.append(pos)
                ds_species_list.append(species)
                ds_energy_list.append(energy) 
                ds_forces_list.append(forces)

        dataset = ANI1xDataset(ds_pos_list,
                               ds_species_list,
                               ds_energy_list,
                               ds_forces_list,
                               cutoff=cutoff
        )
        return dataset
           

    def _collate(self, samples):
        graphs, targets_list = map(list, zip(*samples))
        batched_graph = dgl.batch(graphs)

        # targets
        energy = torch.tensor([targets['energy'] for targets in targets_list])
        forces = torch.cat([targets['forces'] for targets in targets_list])
        targets = {'energy': energy,
                   'forces': forces}

        return batched_graph, targets

    @staticmethod
    def add_argparse_args(parent_parser):
        parser = parent_parser.add_argument_group("ANI1x dataset")
        parser.add_argument('--force_weight', type=float, default=1e-1,
                            help='Weigh force losses to energy losses')
        parser.add_argument('--cutoff', type=float, default=3.0,
                            help='Radius of graph neighborhood')
        return parent_parser

    def __repr__(self):
        return 'ANI1x'

    @staticmethod
    def loss_fn(pred, target):
        energy_loss = F.mse_loss(pred[0], target['energy'])
        forces_loss = F.mse_loss(pred[1], target['forces'])
        return energy_loss, forces_loss



class ANI1xDataset(Dataset):
    def __init__(self, pos_list, species_list, energy_list, forces_list, cutoff=float('inf'), normalize=True, test=False):
        self.pos_list = pos_list
        self.species_list = species_list
        self.energy_list = energy_list
        self.forces_list = forces_list
        self.cutoff = cutoff
        self.normalize = normalize
        self.test = test

        self.species_dict = {1: 0, 6: 1, 7: 2, 8: 3}
        self.test_species_dict = {b'H': 1, b'C': 2, b'N': 3, b'O': 4}

        self.atom_energies = ATOM_ENERGIES
        self.energy_std = ENERGY_STD

    def __len__(self):
        return len(self.pos_list)

    def __getitem__(self, i):
        pos = self.pos_list[i]
        species = self.species_list[i]
        forces = self.forces_list[i]
        energy = self.energy_list[i]

        # Create graph
        pos = torch.tensor(pos)
        graph = self._create_graph(pos, cutoff=self.cutoff)

        # Create node features
        if self.test:
            species = torch.tensor([self.test_species_dict[atom] for atom in species])
        else:
            species = torch.tensor([self.species_dict[atom] for atom in species]) 
        graph.ndata['species'] = species

        # Create targets
        if self.normalize:
            counter = Counter(species.numpy())
            counts = torch.tensor([counter[i] for i in range(NUM_SPECIES)], dtype=torch.float)
            adjustment = counts @ self.atom_energies
            energy = (energy - adjustment) / self.energy_std
            forces = forces / self.energy_std
        forces = torch.tensor(forces)
        targets = {'energy': energy,
                   'forces': forces}

        return graph, targets

    @staticmethod
    def _create_graph(pos, cutoff=float('inf')):
        dist_mat = torch.cdist(pos[None,...], pos[None,...]).squeeze(0)
        adj_mat = (dist_mat < cutoff).fill_diagonal_(False)
        u, v = torch.nonzero(adj_mat, as_tuple=True)
        graph = dgl.graph((u,v), num_nodes=len(pos))
        graph.ndata['pos'] = pos
        return graph
