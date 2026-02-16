import os
import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from torch_geometric.transforms import Compose
# [关键] 必须引入 Data 对象，以便 DensePaddingCollater 识别并进行 Padding
from torch_geometric.data import Data 
import hydra
import numpy as np

# [关键] 引入 La-Proteina 自带的 Padding Collater
from proteinfoundation.utils.dense_padding_data_loader import DensePaddingCollater

class _InnerDataset(Dataset):
    """
    内部使用的 Dataset，负责加载文件 -> 键名映射 -> 转为 PyG Data 对象 -> 应用变换。
    增加了长度过滤以防止 OOM。
    """
    def __init__(self, processed_dir, transforms=None, max_length=800):
        self.processed_dir = processed_dir
        if transforms and isinstance(transforms[0], dict):
             self.transforms = Compose([hydra.utils.instantiate(t) for t in transforms])
        elif transforms:
             self.transforms = Compose(transforms)
        else:
             self.transforms = None

        if not os.path.exists(processed_dir):
            raise ValueError(f"目录 {processed_dir} 不存在")
            
        # 扫描所有 .pt 文件
        all_files = [
            os.path.join(processed_dir, f) 
            for f in os.listdir(processed_dir) 
            if f.endswith('.pt')
        ]
        
        self.data_files = []
        skipped_count = 0

        # [新增] 预读取并过滤长度
        print(f"[Dataset] 正在检查 {len(all_files)} 个文件的序列长度，阈值: {max_length}...")
        for f in all_files:
            try:
                # 简单读取以获取形状信息
                # 如果文件极大，读取可能会慢，但为了防止训练 crash 是值得的
                data = torch.load(f)
                
                # 获取残基数量
                n_res = 0
                if 'coords' in data:
                    n_res = data['coords'].shape[0]
                elif 'all_atom_positions' in data:
                    n_res = data['all_atom_positions'].shape[0]
                
                # 过滤逻辑
                if 0 < n_res <= max_length:
                    self.data_files.append(f)
                else:
                    skipped_count += 1
            except Exception as e:
                print(f"[Dataset] 读取文件 {f} 出错: {e}")

        print(f"[Dataset] 过滤完成: 保留 {len(self.data_files)} 个样本，跳过 {skipped_count} 个过长样本 (>{max_length})")
        
        if not self.data_files:
            raise ValueError(f"在 {processed_dir} 中未找到符合长度要求 (<={max_length}) 的数据文件")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        data_path = self.data_files[idx]
        # 加载原始数据 (dict)
        raw_data = torch.load(data_path)
        
        # --- [关键] 键名映射 (Mapping) ---
        # 1. 坐标: all_atom_positions -> coords
        if 'all_atom_positions' in raw_data:
            raw_data['coords'] = raw_data.pop('all_atom_positions')
            
        # 2. 氨基酸类型: aatype -> residue_type
        if 'aatype' in raw_data:
            raw_data['residue_type'] = raw_data.pop('aatype')
            
        # 3. 原子掩码: all_atom_mask -> coord_mask
        if 'all_atom_mask' in raw_data:
            raw_data['coord_mask'] = raw_data.pop('all_atom_mask')
            
        # 4. 链索引: chain_index -> chains
        if 'chain_index' in raw_data:
            raw_data['chains'] = raw_data.pop('chain_index')

        # 5. 残基编号: residue_index -> residue_pdb_idx
        if 'residue_index' in raw_data:
            raw_data['residue_pdb_idx'] = raw_data.pop('residue_index')

        # --- 封装为 PyG Data 对象 ---
        data = Data(**raw_data)
        
        # 补充 id (用于日志)
        if not hasattr(data, 'id'):
             data.id = os.path.basename(data_path).replace('.pt', '')
             
        # 补充 num_nodes (消除 PyG Warning)
        if not hasattr(data, 'num_nodes') and hasattr(data, 'coords'):
             data.num_nodes = data.coords.shape[0]

        # --- 应用变换 ---
        if self.transforms:
            data = self.transforms(data)
            
        return data

class AntibodySimpleDataset(L.LightningDataModule):
    def __init__(self, processed_dir, batch_size=16, num_workers=8, transforms=None, max_length=800, **kwargs):
        super().__init__()
        self.processed_dir = processed_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.transforms = transforms
        self.max_length = max_length # 接收参数

    def setup(self, stage=None):
        # [修改] 传递 max_length 参数给 InnerDataset
        self.train_dataset = _InnerDataset(self.processed_dir, self.transforms, max_length=self.max_length)
        # 暂时复用训练集做验证
        self.val_dataset = _InnerDataset(self.processed_dir, self.transforms, max_length=self.max_length)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=DensePaddingCollater(self.train_dataset)
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=DensePaddingCollater(self.val_dataset)
        )