import json
import os
import argparse
import pandas as pd
from Bio.PDB import PDBParser
import warnings
from Bio import BiopythonWarning

# 忽略 PDB 解析时常见的一些小警告
warnings.simplefilter('ignore', BiopythonWarning)

# 3字母到单字母氨基酸的映射表
THREE_TO_ONE = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
    'MSE': 'M' # 常见修饰
}

def get_chain_residues(chain):
    """
    提取某条链中所有标准的氨基酸残基
    返回: [(PDB残基编号, 单字母氨基酸), ...]
    """
    residues = []
    for res in chain:
        # res.id 结构为 (hetero_flag, sequence_identifier, insertion_code)
        # 我们只保留标准的氨基酸 (hetero_flag 变为空格)
        if res.id[0] == ' ':
            res_num = res.id[1]
            res_name = THREE_TO_ONE.get(res.get_resname(), 'X')
            if res_name != 'X':
                residues.append((res_num, res_name))
    return residues

def get_contiguous_blocks(res_nums):
    """
    将一串残基编号划分为连续的块。
    例如 [1,2,3, 5,6,7] -> [(1,3), (5,7)]
    """
    if not res_nums: return []
    blocks = []
    start = res_nums[0]
    prev = res_nums[0]
    
    for num in res_nums[1:]:
        # 允许编号相同 (应对可能的插入码情况，如 100A, 100B 编号都是100)
        if num == prev + 1 or num == prev:
            prev = num
        else:
            blocks.append((start, prev))
            start = num
            prev = num
    blocks.append((start, prev))
    return blocks

def format_blocks(chain_id, blocks):
    """将区块格式化为 contig_string 格式，如 ['L1-10', 'L15-30']"""
    parts = []
    for start, end in blocks:
        # La-Proteina / RFdiffusion 格式: 即使是一个氨基酸最好也写 start-end
        parts.append(f"{chain_id}{start}-{end}")
    return parts

def generate_motif_csv(json_path, pdb_dir, output_csv_path, target_cdr="cdrh3"):
    data_list = []
    with open(json_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
                    
    print(f"读取到 {len(data_list)} 条 JSON 数据，开始解析 PDB...")
    csv_rows = []
    parser = PDBParser(QUIET=True)

    for entry in data_list:
        pdb_name = entry.get('pdb', 'unknown')
        
        # 尝试从 entry 里获取 pdb_path，如果没有就用传入的 pdb_dir 拼接
        pdb_path = entry.get('pdb_data_path')
        if not pdb_path or not os.path.exists(pdb_path):
            pdb_path = os.path.join(pdb_dir, f"{pdb_name}.pdb")
            
        if not os.path.exists(pdb_path):
            print(f"[警告] 找不到 PDB: {pdb_path}，跳过。")
            continue

        h_chain_id = entry.get('heavy_chain', 'H')
        l_chain_id = entry.get('light_chain', 'L')
        ag_chain_ids = entry.get('antigen_chains', [])
        cdr_seq = entry.get(f'{target_cdr}_seq', '')

        if not cdr_seq:
            continue

        try:
            structure = parser.get_structure(pdb_name, pdb_path)
            model = structure[0]
        except Exception as e:
            print(f"[警告] 解析 PDB 失败 {pdb_name}: {e}")
            continue

        contig_parts = []
        total_length = 0

        # --- 1. 处理抗原链 (Antigen) ---
        for ag_id in ag_chain_ids:
            if ag_id in model:
                ag_residues = get_chain_residues(model[ag_id])
                ag_nums = [num for num, _ in ag_residues]
                ag_blocks = get_contiguous_blocks(ag_nums)
                contig_parts.extend(format_blocks(ag_id, ag_blocks))
                total_length += len(ag_residues)

        # --- 2. 处理轻链 (Light Chain) ---
        if l_chain_id in model:
            l_residues = get_chain_residues(model[l_chain_id])
            l_nums = [num for num, _ in l_residues]
            l_blocks = get_contiguous_blocks(l_nums)
            contig_parts.extend(format_blocks(l_chain_id, l_blocks))
            total_length += len(l_residues)

        # --- 3. 处理重链 (Heavy Chain) ---
        if h_chain_id not in model:
            print(f"[警告] {pdb_name} 中找不到重链 {h_chain_id}，跳过。")
            continue
            
        h_residues = get_chain_residues(model[h_chain_id])
        h_seq = "".join([aa for _, aa in h_residues])
        h_nums = [num for num, _ in h_residues]

        # 在 PDB 实际提取的序列中查找 CDR
        start_idx = h_seq.find(cdr_seq)
        if start_idx == -1:
            print(f"[警告] {pdb_name} 的 PDB 重链中找不到 CDR 序列 '{cdr_seq}'，跳过。")
            continue
            
        end_idx = start_idx + len(cdr_seq)

        # 前段骨架 (Framework 1)
        fw1_nums = h_nums[:start_idx]
        fw1_blocks = get_contiguous_blocks(fw1_nums)
        contig_parts.extend(format_blocks(h_chain_id, fw1_blocks))
        total_length += len(fw1_nums)

        # 插入生成区间占位符 (CDR-H3)
        h3_len = len(cdr_seq)
        contig_parts.append(f"{h3_len}-{h3_len}")
        total_length += h3_len

        # 后段骨架 (Framework 2)
        fw2_nums = h_nums[end_idx:]
        fw2_blocks = get_contiguous_blocks(fw2_nums)
        contig_parts.extend(format_blocks(h_chain_id, fw2_blocks))
        total_length += len(fw2_nums)

        # 组装最终的 contig string
        contig_str = "/".join(contig_parts)
        
        # 组装 segment_order (A;L;H)
        order_list = []
        for ag_id in ag_chain_ids:
            if ag_id in model and ag_id not in order_list: order_list.append(ag_id)
        if l_chain_id in model and l_chain_id not in order_list: order_list.append(l_chain_id)
        if h_chain_id in model and h_chain_id not in order_list: order_list.append(h_chain_id)
        segment_order = ";".join(order_list)

        csv_rows.append({
            "pdb_name": pdb_name,
            "motif_pdb_path": os.path.abspath(pdb_path),
            "contig_string": contig_str,
            "atom_selection_mode": "all", # 或者 all_atom
            "segment_order": segment_order,
            "total_length": total_length
        })

    # 保存结果
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        df.to_csv(output_csv_path, index=False)
        print(f"\n✅ 成功生成 CSV 文件: {output_csv_path}")
        print(f"✅ 包含样本数: {len(df)}")
        print(f"🔍 示例 {df.iloc[0]['pdb_name']}:")
        print(f"  Contig: {df.iloc[0]['contig_string']}")
        print(f"  Total Length: {df.iloc[0]['total_length']}")
        print(f"  Segment Order: {df.iloc[0]['segment_order']}")
    else:
        print("\n❌ 未生成任何有效数据。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="sabdab_all.json", help="输入的 JSON 文件路径")
    parser.add_argument("--pdb_dir", type=str, default="./pdb", help="PDB 文件夹路径")
    parser.add_argument("--job_id", type=int, default=0, help="Job ID")
    parser.add_argument("--task_name", type=str, default="antibody_inference", help="任务名称")
    
    args = parser.parse_args()
    output_filename = f"{args.task_name}_{args.job_id}_motif_info.csv"
    
    generate_motif_csv(args.json_path, args.pdb_dir, output_filename, target_cdr="cdrh3")