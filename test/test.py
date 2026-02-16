import sys
import os
import torch
from pathlib import Path

# 1. 设置路径，确保能 import proteinfoundation
# 注意：如果这个脚本还在项目内，保留这段逻辑以确保能 import 成功
current_dir = Path(__file__).parent
root_dir = current_dir.parent
sys.path.append(str(root_dir))

from proteinfoundation.datasets.antibody_data import AntibodyDataset
from proteinfoundation.datasets.transforms import AntibodyMotifMaskTransform
from proteinfoundation.datasets.transforms import CoordsToNanometers
from torch_geometric.transforms import Compose

def test_dataloader():
    print(f"Script running from: {current_dir}")
    
    abs_json_path = "/DATA/disk3/yilin/dyMEAN/all_data/test.json" 
    abs_pdb_dir   = "/DATA/disk3/yilin/dyMEAN/all_data/pdb"
    
    # 转换为 Path 对象
    json_path = Path(abs_json_path)
    pdb_dir = Path(abs_pdb_dir)
    
    # 3. 初始化 Dataset
    print("\n--- 1. Loading Raw Data ---")
    dataset = AntibodyDataset(
        json_path=str(json_path),
        pdb_root_dir=str(pdb_dir),
        transform=None
    )
    
    if len(dataset) == 0:
        print("Error: Dataset is empty.")
        return

    graph = dataset[0]
    if graph is None:
        print("Error: Failed to load graph.")
        return

    print(f"Successfully loaded PDB: {graph.id}")
    
    # --- [修改点] 健壮的链信息获取 ---
    # 不要直接用 graph.chain_id，因为它可能不存在
    chains = set()
    if hasattr(graph, 'chain_id'):
        chains = set(graph.chain_id)
    elif hasattr(graph, 'residue_id'):
        # residue_id 格式通常是 "Chain:ResName:ResNum:InsCode"
        # 例如 "A:ARG:39: "
        chains = set(r.split(':')[0] for r in graph.residue_id)
    
    print(f"Chains present: {chains}")
    print(f"Number of residues: {graph.num_nodes}")
    
    # 检查 CDR Mask 是否注入
    if hasattr(graph, 'is_cdr'):
        cdr_count = graph.is_cdr.sum().item()
        print(f"CDR residues count (Masked as True): {cdr_count}")
        print(f"CDR Mask shape: {graph.is_cdr.shape}")
        
        # 打印前 10 个 CDR 残基的索引，方便核对
        cdr_indices = torch.where(graph.is_cdr)[0]
        print(f"Indices of first 10 CDR residues: {cdr_indices[:10].tolist()}")
    else:
        print("Error: 'is_cdr' attribute missing!")

    # 4. 测试 Transform
    print("\n--- 2. Testing Transforms (Motif Generation) ---")
    
    # 模拟训练时的 Transform
    transform_pipeline = Compose([
        CoordsToNanometers(),          # 转纳米
        AntibodyMotifMaskTransform()   # 生成 motif_mask
    ])
    
    # 重新处理 graph
    processed_graph = transform_pipeline(graph)
    print("Transform applied.")
    
    # 检查 motif_mask
    # motif_mask: 1 = Fixed (Framework/Antigen), 0 = Missing (CDR to generate)
    motif_mask = processed_graph.motif_mask
    print(f"Motif Mask shape: {motif_mask.shape}")
    
    # 验证逻辑
    if hasattr(processed_graph, 'is_cdr'):
        cdr_indices = torch.where(processed_graph.is_cdr)[0]
        if len(cdr_indices) > 0:
            test_idx = cdr_indices[0] # 取第一个 CDR 残基
            
            # 检查这个 CDR 残基的 CA 原子 (原子索引1)
            # is_cdr=True -> motif_mask 应该为 0 (False) -> x_motif 应该为 0
            
            is_masked_out = (processed_graph.motif_mask[test_idx, 1] == 0)
            coord_is_zero = (processed_graph.x_motif[test_idx, 1].abs().sum() < 1e-6)
            
            print(f"\nVerification on CDR Residue Index {test_idx}:")
            print(f"  Is CDR? {processed_graph.is_cdr[test_idx]}")
            print(f"  Motif Mask value (should be 0/False): {processed_graph.motif_mask[test_idx, 1]}")
            print(f"  x_motif coordinate (should be 0): {processed_graph.x_motif[test_idx, 1]}")
            
            if is_masked_out and coord_is_zero:
                print(">>> SUCCESS: CDR region is correctly masked out!")
            else:
                print(">>> FAILURE: CDR region was NOT masked correctly.")
    
    print("\n--- Test Finished ---")

if __name__ == "__main__":
    test_dataloader()