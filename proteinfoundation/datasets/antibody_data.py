import json
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from graphein.protein.tensor.io import protein_to_pyg
from openfold.np.residue_constants import resname_to_idx
from proteinfoundation.utils.constants import PDB_TO_OPENFOLD_INDEX_TENSOR
from proteinfoundation.datasets.base_data import BaseLightningDataModule

class AntibodyDataset(Dataset):
    def __init__(self, json_path, pdb_root_dir, transform=None, max_length=None):
        self.pdb_root_dir = Path(pdb_root_dir)
        self.transform = transform
        self.max_length = max_length
        
        with open(json_path, 'r') as f:
            self.data_list = [json.loads(line) for line in f]
            
        print(f"Loaded {len(self.data_list)} antibody entries.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        loop_counter = 0
        max_retries = 10
        fill_value = 1e-5
        
        while loop_counter < max_retries:
            current_idx = (idx + loop_counter) % len(self.data_list)
            entry = self.data_list[current_idx]
            pdb_name = entry['pdb']
            pdb_path = self.pdb_root_dir / f"{pdb_name}.pdb"
            
            chains_to_load = [entry['heavy_chain'], entry['light_chain']] + entry['antigen_chains']
            
            try:
                if not pdb_path.exists():
                    raise FileNotFoundError(f"{pdb_path} not found")

                graph = protein_to_pyg(
                    path=str(pdb_path),
                    chain_selection=chains_to_load,
                    keep_insertions=True,
                    store_het=False,
                    fill_value_coords=fill_value
                )
            except Exception as e:
                loop_counter += 1
                continue

            # Length filtering to prevent OOM
            if self.max_length is not None and graph.coords.shape[0] > self.max_length:
                loop_counter += 1
                continue

            # Data Preprocessing
            if hasattr(graph, 'coords'):
                graph.coord_mask = (graph.coords != fill_value)[..., 0]
                graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
                graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
            
            if graph.num_nodes is None:
                graph.num_nodes = graph.coords.shape[0]
                
            graph.id = pdb_name
            graph.residue_type = torch.tensor(
                [resname_to_idx.get(r, 20) for r in graph.residues]
            ).long()

            self._inject_cdr_mask(graph, entry)

            if self.transform:
                try:
                    graph = self.transform(graph)
                except Exception as e:
                    loop_counter += 1
                    continue
            
            return graph

        return None

    def _inject_cdr_mask(self, graph, entry):
        """
        Masks the CDR-H3 region for generation, fixing all other context.
        """
        num_residues = graph.residue_type.shape[0] if graph.num_nodes is None else graph.num_nodes
        is_cdr = torch.zeros(num_residues, dtype=torch.bool)
        
        chain_ids = []
        res_nums = []
        
        if hasattr(graph, 'chain_id') and hasattr(graph, 'residue_number'):
            chain_ids = graph.chain_id
            res_nums = graph.residue_number
        else:
            for res_str in graph.residue_id:
                parts = res_str.split(':')
                chain_ids.append(parts[0])
                try:
                    res_nums.append(int(parts[2]))
                except ValueError:
                    res_nums.append(-999) 

        # Only retain CDR-H3 from the heavy chain for generation
        cdr_ranges = {
            entry['heavy_chain']: [entry.get('cdrh3_pos')]
        }

        for i in range(num_residues):
            c_id = chain_ids[i]
            r_num = int(res_nums[i])
            
            if c_id in cdr_ranges:
                for rng in cdr_ranges[c_id]:
                    if rng is not None and len(rng) >= 2:
                        if rng[0] <= r_num <= rng[1]:
                            is_cdr[i] = True
                            break
        
        graph.is_cdr = is_cdr

class AntibodyLightningDataModule(BaseLightningDataModule):
    def __init__(self, json_path, pdb_dir, max_length=None, **kwargs):
        super().__init__(**kwargs)
        self.json_path = json_path
        self.pdb_dir = pdb_dir
        self.max_length = max_length

    def _get_dataset(self, split):
        return AntibodyDataset(
            self.json_path, 
            self.pdb_dir, 
            transform=self.transform, 
            max_length=self.max_length
        )