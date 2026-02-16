import json
import pandas as pd
import os
import argparse

def generate_motif_csv(json_path, output_csv_path, target_cdr="cdrh3"):
    """
    根据 SabDab JSON 生成 La-Proteina 推理所需的 motif_info.csv
    """
    
    # 1. 读取 JSON 数据
    # 如果文件是一行一个 JSON 对象 (JSONL 格式)
    data_list = []
    with open(json_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data_list.append(json.loads(line))
                except json.JSONDecodeError:
                    pass # 跳过空行或错误行
    
    # 如果文件是标准的 JSON 列表格式 [{}, {}]，请取消下面这行的注释并注释掉上面的循环
    # with open(json_path, 'r') as f: data_list = json.load(f)

    print(f"读取到 {len(data_list)} 条数据。")

    csv_rows = []

    for entry in data_list:
        pdb_name = entry.get('pdb', 'unknown')
        pdb_path = entry.get('pdb_data_path')
        
        # 确保 PDB 路径存在 (可选检查)
        if not os.path.exists(pdb_path):
            print(f"[警告] PDB 文件未找到: {pdb_path}，跳过。")
            continue

        # 获取链 ID
        h_chain_id = entry.get('heavy_chain', 'H')
        l_chain_id = entry.get('light_chain', 'L')
        ag_chain_ids = entry.get('antigen_chains', [])

        # 获取序列
        h_seq = entry.get('heavy_chain_seq', '')
        l_seq = entry.get('light_chain_seq', '')
        
        # 获取 CDR 序列 (用于定位)
        cdr_seq = entry.get(f'{target_cdr}_seq', '')
        
        if not h_seq or not cdr_seq:
            print(f"[跳过] {pdb_name} 缺少重链或 CDR 序列信息。")
            continue

        # --- 核心逻辑：通过序列匹配定位 Mask 区域 ---
        # 我们需要在 h_seq 中找到 cdr_seq 的位置
        # 注意：这里假设 JSON 中的 seq 与 PDB 文件中的残基是 1:1 对应的连续索引
        start_idx = h_seq.find(cdr_seq)
        
        if start_idx == -1:
            print(f"[跳过] {pdb_name} 重链序列中未找到 CDR 序列。")
            continue
            
        end_idx = start_idx + len(cdr_seq)
        
        # 转换为 1-based 索引 (La-Proteina/PDB 通常使用 1-based)
        # Python: 0...start_idx-1 (是前段), start_idx...end_idx-1 (是CDR), end_idx... (是后段)
        # 1-based:
        #   前段骨架: 1 到 start_idx
        #   CDR: start_idx+1 到 end_idx (我们要 Mask 掉这部分)
        #   后段骨架: end_idx+1 到 len(h_seq)
        
        fixed_parts = []
        
        # 1. 重链骨架 (Mask 掉 CDR)
        # 如果 CDR 在最开头 (不太可能)，则没有前段
        if start_idx > 0:
            fixed_parts.append(f"{h_chain_id}1-{start_idx}")
        
        # 如果 CDR 在最末尾，则没有后段
        if end_idx < len(h_seq):
            fixed_parts.append(f"{h_chain_id}{end_idx + 1}-{len(h_seq)}")
            
        # 2. 轻链 (全部固定)
        if l_seq:
            fixed_parts.append(f"{l_chain_id}1-{len(l_seq)}")
        
        # 3. 抗原链 (全部固定)
        for ag_chain in ag_chain_ids:
            # 抗原长度未知，但在 La-Proteina 中，如果只写链名 (例如 "A")，通常表示整条链
            # 或者我们需要读取 PDB 获取长度。为了安全，如果 generate.py 支持 "A"，则用 "A"
            # 如果不支持，可以尝试读取 PDB。
            # 根据您之前的 generate.py，它支持解析 contig string。
            # 最稳妥的方式是：如果知道长度最好，如果不知道，La-Proteina 的 Dataset 可能需要处理。
            # 这里我们假设简单的 "ChainID" 格式被支持，或者我们假设抗原保留所有原子。
            fixed_parts.append(f"{ag_chain}") 

        # 构造 contig_string
        contig_str = "/".join(fixed_parts)
        
        csv_rows.append({
            "pdb_name": pdb_name,
            "motif_pdb_path": pdb_path,
            "contig_string": contig_str,
            "atom_selection_mode": "all" # 或者 "backbone"，取决于你想给模型看什么
        })

    # 保存 CSV
    if csv_rows:
        df = pd.DataFrame(csv_rows)
        df.to_csv(output_csv_path, index=False)
        print(f"成功生成 CSV 文件: {output_csv_path}")
        print(f"包含样本数: {len(df)}")
        print("示例 Contig String:", df.iloc[0]['contig_string'])
    else:
        print("未生成任何有效数据。")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--json_path", type=str, default="sabdab_all.json", help="输入的 JSON 文件路径")
    parser.add_argument("--job_id", type=int, default=0, help="Job ID，用于生成对应的文件名")
    parser.add_argument("--task_name", type=str, default="antibody_test", help="任务名称")
    
    args = parser.parse_args()
    
    # 构造输出文件名：{task_name}_{job_id}_motif_info.csv
    output_filename = f"{args.task_name}_{args.job_id}_motif_info.csv"
    
    generate_motif_csv(args.json_path, output_filename, target_cdr="cdrh3")