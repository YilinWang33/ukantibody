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
        """
        Args:
            json_path: sabdab_all.json 的路径
            pdb_root_dir: 存放 PDB 文件的根目录
            transform: PyG 变换 (transforms)
            max_length: [新增] 最大序列长度限制，超过此长度的样本将被跳过 (防止 OOM)
        """
        self.pdb_root_dir = Path(pdb_root_dir)
        self.transform = transform
        self.max_length = max_length
        
        # 读取 JSON 数据
        with open(json_path, 'r') as f:
            self.data_list = [json.loads(line) for line in f]
            
        print(f"Loaded {len(self.data_list)} antibody entries.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # 使用循环机制，如果当前样本加载失败或太长，自动尝试下一个
        # 避免训练因单个坏数据而中断
        loop_counter = 0
        max_retries = 10
        
        while loop_counter < max_retries:
            # 获取当前索引的数据条目
            # 注意：如果发生了重试，idx 需要更新到下一个
            current_idx = (idx + loop_counter) % len(self.data_list)
            entry = self.data_list[current_idx]
            pdb_name = entry['pdb']
            
            pdb_path = self.pdb_root_dir / f"{pdb_name}.pdb"
            
            # 定义需要加载的链：重链 + 轻链 + 抗原链
            chains_to_load = [entry['heavy_chain'], entry['light_chain']] + entry['antigen_chains']
            
            # 填充值，用于标记缺失原子
            fill_value = 1e-5
            
            try:
                # 尝试加载 PDB
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
                # 加载失败，打印错误并重试
                print(f"[Warning] Error loading {pdb_name}: {e}. Skipping...")
                loop_counter += 1
                continue

            # --- [关键修改] 长度检查防止 OOM ---
            # graph.coords 维度通常是 [N, 37, 3]
            current_length = graph.coords.shape[0]
            if self.max_length is not None and current_length > self.max_length:
                print(f"[Skipping] {pdb_name}: length {current_length} > {self.max_length}")
                loop_counter += 1
                continue

            # --- 数据预处理 ---
            
            # 1. 生成 coord_mask (有效原子掩码)
            if hasattr(graph, 'coords'):
                # 坐标不等于填充值的地方即为有效原子
                graph.coord_mask = (graph.coords != fill_value)[..., 0]

                # 2. [非常重要] 将 PDB 原子顺序重排为 OpenFold 标准顺序
                graph.coords = graph.coords[:, PDB_TO_OPENFOLD_INDEX_TENSOR, :]
                graph.coord_mask = graph.coord_mask[:, PDB_TO_OPENFOLD_INDEX_TENSOR]
            
            # 3. [修复 Warning] 显式设置 num_nodes
            if graph.num_nodes is None:
                graph.num_nodes = graph.coords.shape[0]
                
            # 4. 设置 ID
            graph.id = pdb_name
            
            # 5. 序列编码 (Residue Type)
            # resname_to_idx 将 'ALA' -> 0 等，未知残基给 20
            graph.residue_type = torch.tensor(
                [resname_to_idx.get(r, 20) for r in graph.residues]
            ).long()

            # 6. 注入 CDR 掩码 (区分 Motif 和生成区域)
            self._inject_cdr_mask(graph, entry)

            # 7. 应用 Transforms (如 CoordsToNanometers, MotifMaskTransform)
            if self.transform:
                try:
                    graph = self.transform(graph)
                except Exception as e:
                    print(f"[Warning] Error applying transform to {pdb_name}: {e}. Skipping...")
                    loop_counter += 1
                    continue
            
            # 成功处理，返回图对象
            return graph

        # 如果重试了多次都失败，返回 None (DataLoader 的 collate_fn 需要能处理 None)
        # 或者抛出异常
        print(f"[Error] Failed to load valid data after {max_retries} attempts starting from idx {idx}")
        return None
    '''
    def _inject_cdr_mask(self, graph, entry):
        """
        根据 JSON 中的 CDR pos (IMGT编号) 生成 is_cdr 掩码。
        is_cdr = True  -> CDR 区域 (Diffused / Generated)
        is_cdr = False -> Framework / Antigen (Fixed Motif)
        """
        # 确保 num_nodes 存在
        if graph.num_nodes is None:
            num_residues = graph.residue_type.shape[0]
            graph.num_nodes = num_residues
        else:
            num_residues = graph.num_nodes
            
        is_cdr = torch.zeros(num_residues, dtype=torch.bool)
        
        # 解析链 ID 和残基编号
        chain_ids = []
        res_nums = []
        
        if hasattr(graph, 'chain_id') and hasattr(graph, 'residue_number'):
            # Graphein 标准属性
            chain_ids = graph.chain_id
            res_nums = graph.residue_number
        else:
            # 备用方案：从 residue_id 字符串解析 (格式 "Chain:ResName:ResNum:InsCode")
            for res_str in graph.residue_id:
                parts = res_str.split(':')
                chain_ids.append(parts[0])
                try:
                    res_nums.append(int(parts[2]))
                except ValueError:
                    # 处理可能的插入码情况或非数字
                    res_nums.append(-999) 

        # 获取 CDR 范围定义
        cdr_ranges = {
            entry['heavy_chain']: [
                entry.get('cdrh1_pos'), entry.get('cdrh2_pos'), entry.get('cdrh3_pos')
            ],
            entry['light_chain']: [
                entry.get('cdrl1_pos'), entry.get('cdrl2_pos'), entry.get('cdrl3_pos')
            ]
        }

        # 遍历所有残基，标记 CDR
        for i in range(num_residues):
            c_id = chain_ids[i]
            r_num = int(res_nums[i])
            
            if c_id in cdr_ranges:
                for rng in cdr_ranges[c_id]:
                    # rng 格式通常是 [start, end]
                    if rng is not None and len(rng) >= 2:
                        if rng[0] <= r_num <= rng[1]:
                            is_cdr[i] = True
                            break
        
        graph.is_cdr = is_cdr
    '''
    def _inject_cdr_mask(self, graph, entry):
        """
        根据 JSON 中的 CDR pos (IMGT编号) 生成 is_cdr 掩码。
        修改版：仅仅 Mask CDR-H3 区域，固定其余所有上下文！
        """
        if graph.num_nodes is None:
            num_residues = graph.residue_type.shape[0]
            graph.num_nodes = num_residues
        else:
            num_residues = graph.num_nodes
            
        is_cdr = torch.zeros(num_residues, dtype=torch.bool)
        
        # 解析链 ID 和残基编号
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

        # [关键修改] 仅保留 heavy_chain 的 cdrh3_pos
        cdr_ranges = {
            entry['heavy_chain']: [
                entry.get('cdrh3_pos')
            ]
        }

        # 遍历所有残基，标记 CDR-H3
        for i in range(num_residues):
            c_id = chain_ids[i]
            r_num = int(res_nums[i])
            
            if c_id in cdr_ranges:
                for rng in cdr_ranges[c_id]:
                    # rng 格式通常是 [start, end]
                    if rng is not None and len(rng) >= 2:
                        if rng[0] <= r_num <= rng[1]:
                            is_cdr[i] = True
                            break
        
        graph.is_cdr = is_cdr
# 这是一个 LightningDataModule，用于连接 Trainer
class AntibodyLightningDataModule(BaseLightningDataModule):
    def __init__(self, json_path, pdb_dir, max_length=None, **kwargs):
        super().__init__(**kwargs)
        self.json_path = json_path
        self.pdb_dir = pdb_dir
        self.max_length = max_length # [新增] 保存参数

    def _get_dataset(self, split):
        # 2. 传递给 Dataset
        return AntibodyDataset(
            self.json_path, 
            self.pdb_dir, 
            transform=self.transform, 
            max_length=self.max_length
        )