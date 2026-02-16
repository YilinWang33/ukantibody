import torch
import numpy as np
import os
# 从你的项目中导入刚刚定义好的类
from proteinfoundation.datasets.antibody_simple_dataset import AntibodySimpleDataset

def test_antibody_dataset():
    # 1. 设定数据路径
    processed_dir = "./data/antibody_processed" 
    
    print(f"正在尝试从 {processed_dir} 加载数据集...")
    
    try:
        # 实例化数据集
        dataset = AntibodySimpleDataset(processed_dir=processed_dir)
    except ValueError as e:
        print(f"加载失败: {e}")
        print("请检查路径是否正确，或运行预处理脚本生成 .pt 文件。")
        return

    print(f"数据集大小: {len(dataset)}")
    
    if len(dataset) == 0:
        print("警告: 数据集为空！")
        return

    # 2. 获取第一个样本
    sample = dataset[0]
    print("\n成功加载第一个样本！")
    print("-" * 30)
    
    # 3. 检查关键字段
    keys_to_check = [
        "aatype", 
        "all_atom_positions", 
        "all_atom_mask", 
        "rigidgroups_gt_frames", 
        "fixed_mask"
    ]
    
    for key in keys_to_check:
        if key in sample:
            val = sample[key]
            if isinstance(val, torch.Tensor):
                print(f"{key}: Shape={val.shape}, Type={val.dtype}")
            else:
                print(f"{key}: {type(val)}")
        else:
            print(f"错误: 缺少关键字段 '{key}'")

    # 4. 逻辑检查
    if "fixed_mask" in sample:
        fixed_mask = sample["fixed_mask"]
        num_cdr_residues = (fixed_mask == 0.0).sum().item()
        print(f"\nCDR 残基数量 (fixed_mask == 0): {num_cdr_residues}")
        
        if num_cdr_residues == 0:
            print("警告: fixed_mask 全为 1，没有标记 CDR 区域！")
        else:
            print("Mask 检查通过：包含需生成区域。")

    # 5. 检查刚体坐标系 (Rigid Frames)
    if "rigidgroups_gt_frames" in sample:
        rigids = sample["rigidgroups_gt_frames"]
        if rigids.shape[-2:] == (4, 4):
            print("Rigid Frames 形状正确。")
        else:
            print(f"Rigid Frames 形状错误: {rigids.shape}")

if __name__ == "__main__":
    test_antibody_dataset()