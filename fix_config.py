import json
import argparse
import numpy as np
from Bio.PDB import PDBParser, Polypeptide

def generate_yaml_from_json(data):
    """
    根据输入的 JSON 数据和对应的 PDB 文件，自动生成绝对安全的 YAML 配置。
    """
    pdb_path = data['pdb_data_path']
    heavy_chain = data['heavy_chain']
    light_chain = data.get('light_chain', '')
    antigen_chains = data.get('antigen_chains', [])
    
    # 默认针对 CDR-H3 进行设计，你可以根据需要修改
    target_chain = heavy_chain
    target_cdr_seq = data['cdrh3_seq']
    target_cdr_length = len(target_cdr_seq)

    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein', pdb_path)
    model = structure[0]

    # 按顺序处理链：抗原 -> 轻链 -> 重链 (严格按照模型期望的顺序)
    chains_to_process = []
    for achain in antigen_chains:
        if achain in model: chains_to_process.append(achain)
    if light_chain and light_chain in model: chains_to_process.append(light_chain)
    if heavy_chain in model: chains_to_process.append(heavy_chain)

    contigs = []
    total_extracted_length = 0

    for chain_id in chains_to_process:
        chain = model[chain_id]
        
        valid_res = []
        seq = ""
        # 提取有 CA 原子的真实氨基酸（排除水和杂原子）
        for res in chain:
            if 'CA' in res and res.id[0] == ' ':
                try:
                    seq += Polypeptide.three_to_one(res.resname)
                    valid_res.append(res)
                except:
                    seq += "X"
                    valid_res.append(res)
        
        if not valid_res:
            continue

        cdr_res_ids = set()
        if chain_id == target_chain:
            # 在序列中精准寻找 CDR 序列，确保 100% 定位
            idx = seq.find(target_cdr_seq)
            if idx != -1:
                cdr_res_ids = set([r.id[1] for r in valid_res[idx : idx + len(target_cdr_seq)]])
            else:
                print(f"[警告] 在链 {chain_id} 中未精确找到序列 '{target_cdr_seq}'，退回使用 JSON 索引...")
                start, end = data['cdrh3_pos']
                cdr_res_ids = set([r.id[1] for r in valid_res[start:end+1]])
        
        # 抠掉 CDR 后剩下的框架残基
        non_cdr_res = [r for r in valid_res if r.id[1] not in cdr_res_ids]
        
        if not non_cdr_res:
            if chain_id == target_chain:
                contigs.append(f"{target_cdr_length}-{target_cdr_length}")
                total_extracted_length += target_cdr_length
            continue

        chain_contig_parts = []
        
        # 检查待生成的 CDR 是否在链的最头部
        if chain_id == target_chain and cdr_res_ids and non_cdr_res[0].id[1] > max(cdr_res_ids):
            chain_contig_parts.append(f"{target_cdr_length}-{target_cdr_length}")
            total_extracted_length += target_cdr_length

        current_block = [non_cdr_res[0]]

        for i in range(1, len(non_cdr_res)):
            prev = non_cdr_res[i-1]
            curr = non_cdr_res[i]

            # 计算真实物理距离 (用于判断晶体结构缺失)
            dist = np.linalg.norm(prev['CA'].get_coord() - curr['CA'].get_coord())
            # 检查编号是否连续 (用于处理 IMGT 的跳号问题)
            is_num_jump = (curr.id[1] != prev.id[1] + 1)
            
            # 检查两人之间是不是正好是被我们抠掉的 CDR
            is_cdr_skipped = False
            if chain_id == target_chain and cdr_res_ids:
                if prev.id[1] < min(cdr_res_ids) and curr.id[1] > max(cdr_res_ids):
                    is_cdr_skipped = True

            # 出现任何不连续：CDR占位 / 物理断开 / 编号跳跃
            if is_cdr_skipped or (dist > 4.5):
                start_num = current_block[0].id[1]
                end_num = current_block[-1].id[1]
                
                if start_num == end_num:
                    chain_contig_parts.append(f"{chain_id}{start_num}")
                else:
                    chain_contig_parts.append(f"{chain_id}{start_num}-{end_num}")
                
                total_extracted_length += len(current_block)

                if is_cdr_skipped:
                    # 填入我们要生成的 CDR
                    chain_contig_parts.append(f"{target_cdr_length}-{target_cdr_length}")
                    total_extracted_length += target_cdr_length
                elif dist > 4.5 and not is_cdr_skipped:
                    # 真实物理空间断开，告诉模型需要用生成方式缝合
                    gap_len = curr.id[1] - prev.id[1] - 1
                    if gap_len <= 0: gap_len = 1
                    chain_contig_parts.append(f"{gap_len}-{gap_len}")
                    total_extracted_length += gap_len
                # 如果仅仅是 IMGT 编号跳跃（距离正常 < 4.5），直接截断片段但不插数字！

                current_block = [curr]
            else:
                current_block.append(curr)

        # 结算每条链的最后一块
        if current_block:
            start_num = current_block[0].id[1]
            end_num = current_block[-1].id[1]
            if start_num == end_num:
                chain_contig_parts.append(f"{chain_id}{start_num}")
            else:
                chain_contig_parts.append(f"{chain_id}{start_num}-{end_num}")
            total_extracted_length += len(current_block)

        # 检查待生成的 CDR 是否在链的最尾部
        if chain_id == target_chain and cdr_res_ids and non_cdr_res[-1].id[1] < min(cdr_res_ids):
            chain_contig_parts.append(f"{target_cdr_length}-{target_cdr_length}")
            total_extracted_length += target_cdr_length

        contigs.append("/".join(chain_contig_parts))

    # 组装最终结果
    contig_string = "/".join(contigs)
    segment_order = ";".join(chains_to_process)
    pdb_name = data.get('pdb', 'unknown_target')
    
    yaml_output = f"""
      {pdb_name}_cdrh3:
        contig_string: "{contig_string}"
        motif_pdb_path: "{pdb_path}"
        atom_selection_mode: "all_atom"
        motif_only: False
        motif_min_length: {total_extracted_length}
        motif_max_length: {total_extracted_length}
        segment_order: "{segment_order}"
"""
    print(f"\n======== [{pdb_name}] 生成的 YAML 配置 ========")
    print(yaml_output.strip('\n'))
    print("==================================================\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动读取抗体 JSON，分析 PDB 并生成扩散模型 YAML 采样配置")
    parser.add_argument("json_path", type=str, help="输入的 JSON 文件路径 (如: ./data/5mes.json)")
    args = parser.parse_args()

    try:
        with open(args.json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        generate_yaml_from_json(data)
    except FileNotFoundError:
        print(f"[错误] 找不到 JSON 文件: {args.json_path}")
    except json.JSONDecodeError:
        print(f"[错误] JSON 文件格式不正确，请检查: {args.json_path}")
    except Exception as e:
        print(f"[错误] 运行过程中发生异常: {e}")