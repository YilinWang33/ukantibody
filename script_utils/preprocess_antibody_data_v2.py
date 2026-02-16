import os
import torch
import numpy as np
import json
from Bio.PDB import PDBParser, Polypeptide
from tqdm import tqdm
import argparse

# --- 1. 定义常量 (标准 AlphaFold/OpenFold 原子映射) ---
# 这是一个标准的 37 原子顺序，LD4 模型严格依赖此顺序
atom_types = [
    "N", "CA", "C", "CB", "O", "CG", "CG1", "CG2", "OG", "OG1", "SG", "CD",
    "CD1", "CD2", "ND1", "ND2", "OD1", "OD2", "SD", "CE", "CE1", "CE2", "CE3",
    "NE", "NE1", "NE2", "OE1", "OE2", "CH2", "NH1", "NH2", "OH", "CZ", "CZ2",
    "CZ3", "NZ", "OXT"
]
atom_order = {atom_type: i for i, atom_type in enumerate(atom_types)}

# 三字母转一字母
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'UNK': 'X'
}

# 氨基酸转数字索引
AA_MAP = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
    'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20, '-': 20
}

def get_full_atom_coords(residue):
    """
    [关键修正] 提取全原子坐标 (37, 3)，而不仅仅是 Backbone。
    这对于 LD4 全原子模型至关重要。
    """
    coords = np.zeros((37, 3))
    mask = np.zeros(37)
    
    for atom in residue:
        name = atom.get_name()
        # 处理一些常见的命名差异
        if name == 'SE' and residue.get_resname() == 'MSE': name = 'SD'
        
        if name in atom_order:
            idx = atom_order[name]
            coords[idx] = atom.get_coord()
            mask[idx] = 1.0
            
    return coords, mask

def process_single_pdb(pdb_path, h_chain_id, l_chain_id, ag_chain_ids, cdr_seq):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('struct', pdb_path)
    except Exception as e:
        print(f"Error parsing {pdb_path}: {e}")
        return None
    model = structure[0]
    
    all_coords, all_seq, all_mask, all_chain_idx, all_res_idx = [], [], [], [], []
    
    # 定义处理顺序：重链 -> 轻链 -> 抗原链
    # 这样 CDR 通常会在序列的前半部分，便于 debug
    chain_info = []
    if h_chain_id in model: chain_info.append({'id': h_chain_id, 'type': 'heavy'})
    if l_chain_id and l_chain_id in model: chain_info.append({'id': l_chain_id, 'type': 'light'})
    if ag_chain_ids:
        for ag_id in ag_chain_ids:
            if ag_id in model: chain_info.append({'id': ag_id, 'type': 'antigen'})

    current_residue_offset = 0  # 累加器，用于处理链间 index 跳跃
    cdr_found = False
    global_seq_idx = 0 # 全局序列索引，用于标记 CDR 在 concat 后的位置
    cdr_range = (-1, -1) # (start, end)

    for chain_cfg in chain_info:
        cid = chain_cfg['id']
        chain_type = chain_cfg['type']
        chain_obj = model[cid]
        
        # 获取标准氨基酸
        residues = [r for r in chain_obj if Polypeptide.is_aa(r, standard=True) or r.get_resname() == 'MSE']
        if not residues: continue

        c_coords, c_seq, c_mask = [], [], []
        
        # 临时存储序列字符串用于匹配 CDR
        c_seq_str = ""

        for res in residues:
            res_name_3 = res.get_resname()
            res_name_1 = THREE_TO_ONE.get(res_name_3, 'X')
            
            # 1. 提取全原子坐标
            coords, mask = get_full_atom_coords(res)
            
            c_coords.append(coords)
            c_mask.append(mask)
            c_seq.append(AA_MAP.get(res_name_1, 20))
            c_seq_str += res_name_1
        
        n_res = len(c_seq)
        
        # 2. 定位 CDR (仅在重链中查找)
        if chain_type == 'heavy' and cdr_seq and not cdr_found:
            idx = c_seq_str.find(cdr_seq)
            if idx != -1:
                cdr_start = global_seq_idx + idx
                cdr_end = cdr_start + len(cdr_seq)
                cdr_range = (cdr_start, cdr_end)
                cdr_found = True
        
        # 3. 构建特征
        all_coords.append(np.array(c_coords))
        all_mask.append(np.array(c_mask))
        all_seq.append(np.array(c_seq))
        
        # 链索引 (Heavy=0, Light=1, Antigen=2...)
        # 这里简化处理：抗体=0, 抗原=1 也可以，或者给每个链唯一 ID
        # 也可以直接用 enumerate 的 index
        chain_idx_val = 0 if chain_type in ['heavy', 'light'] else 1
        all_chain_idx.append(np.full(n_res, chain_idx_val))

        # [关键修正] 构建 Residue Index 并插入跳跃
        # 这里的 +1000 保证了链之间在空间 Attention 上是断开的
        chain_res_idx = np.arange(1, n_res + 1) + current_residue_offset
        all_res_idx.append(chain_res_idx)
        
        # 更新 Offset (链长 + 跳跃值)
        current_residue_offset += n_res + 200 
        global_seq_idx += n_res

    if not all_coords: return None
    # 如果必须要有 CDR 但没找到，返回 None (取决于你的需求)
    if cdr_seq and not cdr_found: 
        # print(f"Skipping {pdb_path}: CDR sequence not found in Heavy Chain.")
        return None

    # 合并所有链
    cat_coords = np.concatenate(all_coords, axis=0) # (L, 37, 3)
    cat_mask = np.concatenate(all_mask, axis=0)     # (L, 37)
    cat_seq = np.concatenate(all_seq, axis=0)       # (L,)
    cat_chain_idx = np.concatenate(all_chain_idx, axis=0)
    cat_res_idx = np.concatenate(all_res_idx, axis=0)


    # [关键修正] 中心化 (Centering)
    # 仅使用存在的原子 (mask=1) 计算中心
    valid_atom_mask = cat_mask.astype(bool)
    if valid_atom_mask.sum() > 0:
        center = cat_coords[valid_atom_mask].mean(axis=0)
        cat_coords = cat_coords - center

    # [关键修正] 构建 motif_mask (用于 Loss Masking)
    # La-Proteina 中，motif_mask=1 表示固定/已知部分(Condition)，motif_mask=0 表示需要生成部分(Inpainting)
    motif_mask = np.ones(len(cat_seq), dtype=np.float32) # 默认为 1 (全部固定)
    if cdr_found:
        # 将 CDR 区域设为 0 (需要生成)
        motif_mask[cdr_range[0]:cdr_range[1]] = 0.0

    return {
        "aatype": torch.tensor(cat_seq, dtype=torch.long),
        "all_atom_positions": torch.tensor(cat_coords, dtype=torch.float32),
        "all_atom_mask": torch.tensor(cat_mask, dtype=torch.float32),
        "chain_index": torch.tensor(cat_chain_idx, dtype=torch.long),
        "residue_index": torch.tensor(cat_res_idx, dtype=torch.long),
        "motif_mask": torch.tensor(motif_mask, dtype=torch.float32) # 改名为 motif_mask 以匹配意图
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to JSONL summary")
    parser.add_argument("--pdb_dir", required=True, help="Path to PDB directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for .pt files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    entries = []
    with open(args.json_path, 'r') as f:
        for line in f:
            if line.strip(): entries.append(json.loads(line))
    
    count = 0
    for entry in tqdm(entries, desc="Processing"):
        pdb_id = entry['pdb']
        pdb_path = os.path.join(args.pdb_dir, f"{pdb_id}.pdb")
        
        if not os.path.exists(pdb_path): continue
            
        data = process_single_pdb(
            pdb_path, 
            entry['heavy_chain'], 
            entry['light_chain'], 
            entry.get('antigen_chains', []), 
            entry['cdrh3_seq']
        )
        
        if data:
            torch.save(data, os.path.join(args.out_dir, f"{pdb_id}.pt"))
            count += 1
            
    print(f"Successfully processed {count} samples.")

if __name__ == "__main__":
    main()