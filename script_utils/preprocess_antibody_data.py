import os
import torch
import numpy as np
import json
from Bio.PDB import PDBParser, Polypeptide
from tqdm import tqdm
import argparse

# --- 新增：手动定义三字母到单字母的映射 ---
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M', 'UNK': 'X'  # 处理常见的 MSE (硒代蛋氨酸) 和 UNK
}

# 简化的氨基酸映射 (保持不变)
AA_MAP = {
    'A': 0, 'R': 1, 'N': 2, 'D': 3, 'C': 4, 'Q': 5, 'E': 6, 'G': 7,
    'H': 8, 'I': 9, 'L': 10, 'K': 11, 'M': 12, 'F': 13, 'P': 14,
    'S': 15, 'T': 16, 'W': 17, 'Y': 18, 'V': 19, 'X': 20, '-': 20
}

def get_atom_coords(residue):
    """提取 N, CA, C, O 坐标，填充到 37 个原子位中"""
    coords = np.zeros((37, 3))
    atom_names = ['N', 'CA', 'C', 'O']
    mask = np.zeros(37)
    
    has_backbone = True
    for i, name in enumerate(atom_names):
        if name in residue:
            coords[i] = residue[name].get_coord()
            mask[i] = 1.0
        else:
            has_backbone = False
            
    return coords, mask, has_backbone

def parse_chain(chain):
    coords_list, seq_list, mask_list, res_idx_list = [], [], [], []
    
    # 过滤非氨基酸
    residues = [r for r in chain if Polypeptide.is_aa(r, standard=True)]
    
    for res in residues:
        coords, mask, has_backbone = get_atom_coords(res)
        if has_backbone:
            coords_list.append(coords)
            mask_list.append(mask)
            
            # --- 修改开始：使用自定义字典进行映射，解决报错 ---
            res_name_3 = res.get_resname()
            # 如果找不到三字母代码，默认为 'X'
            res_name_1 = THREE_TO_ONE.get(res_name_3, 'X')
            # 转换为索引
            seq_list.append(AA_MAP.get(res_name_1, 20))
            # --- 修改结束 ---
            
            res_idx_list.append(res.id[1]) # PDB Residue Number
            
    return np.array(coords_list), np.array(seq_list), np.array(mask_list), np.array(res_idx_list)

def process_single_pdb(pdb_path, h_chain, l_chain, ag_chains, cdr_seq):
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure('struct', pdb_path)
    except Exception: return None
    model = structure[0]
    
    all_coords, all_seq, all_mask, all_chain_idx, all_res_idx = [], [], [], [], []
    
    # H -> L -> Ag 顺序处理
    chains = [(h_chain, 0)]
    if l_chain and l_chain in model: 
        chains.append((l_chain, 1))
    
    # 处理抗原链列表
    if ag_chains:
        for i, ag in enumerate(ag_chains):
            if ag in model: 
                chains.append((ag, 2 + i))
            
    current_len = 0
    cdr_start, cdr_end = -1, -1
    
    for cid, cidx in chains:
        if cid not in model: continue
        c_coords, c_seq, c_mask, c_res_idx = parse_chain(model[cid])
        if len(c_seq) == 0: continue
        
        # 查找 CDR 位置 (仅在重链)
        if cidx == 0:
            # 反向查找 AA_MAP 还原序列字符串用于匹配
            # 注意：这里稍微有些低效，但为了保持逻辑一致性不做大改
            keys = list(AA_MAP.keys())
            vals = list(AA_MAP.values())
            h_seq = "".join([keys[vals.index(i)] for i in c_seq])
            
            # 序列匹配定位 CDR
            if cdr_seq in h_seq:
                start = h_seq.find(cdr_seq)
                cdr_start = current_len + start
                cdr_end = cdr_start + len(cdr_seq)
            else: 
                # 如果找不到完全匹配，可能是编号问题或突变，暂时跳过
                return None 

        all_coords.append(c_coords)
        all_seq.append(c_seq)
        all_mask.append(c_mask)
        all_chain_idx.append(np.full(len(c_seq), cidx))
        all_res_idx.append(c_res_idx)
        current_len += len(c_seq)

    if not all_coords or cdr_start == -1: return None

    cat_coords = np.concatenate(all_coords, axis=0)
    cat_seq = np.concatenate(all_seq, axis=0)
    
    # Fixed Mask: 1.0 = Fixed (Context), 0.0 = Generate (CDR)
    fixed_mask = np.ones(len(cat_seq), dtype=np.float32)
    fixed_mask[cdr_start:cdr_end] = 0.0

    return {
        "aatype": torch.tensor(cat_seq, dtype=torch.long),
        "all_atom_positions": torch.tensor(cat_coords, dtype=torch.float32),
        "all_atom_mask": torch.tensor(np.concatenate(all_mask, axis=0), dtype=torch.float32),
        "chain_index": torch.tensor(np.concatenate(all_chain_idx, axis=0), dtype=torch.long),
        "residue_index": torch.tensor(np.concatenate(all_res_idx, axis=0), dtype=torch.long),
        "fixed_mask": torch.tensor(fixed_mask, dtype=torch.float32)
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", required=True, help="Path to summary.json (JSONL format)")
    parser.add_argument("--pdb_dir", required=True, help="Path to PDB directory")
    parser.add_argument("--out_dir", required=True, help="Output directory for .pt files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    
    # 读取 JSONL 文件
    entries = []
    with open(args.json_path, 'r') as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    
    count = 0
    # 使用 tqdm 显示进度条
    for entry in tqdm(entries, desc="Processing PDBs"):
        pdb_id = entry['pdb']
        # 假设 pdb_dir 下的文件名为 {pdb_id}.pdb
        pdb_path = os.path.join(args.pdb_dir, f"{pdb_id}.pdb")
        
        if not os.path.exists(pdb_path): 
            continue
            
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
            
    print(f"Processed {count} antibodies.")

if __name__ == "__main__":
    main()