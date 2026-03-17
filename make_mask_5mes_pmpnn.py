import json
import os

def create_fixed_positions(pdb_path, target_chain, design_start, design_end, output_file="fixed_positions.jsonl"):
    pdb_name = os.path.basename(pdb_path).replace(".pdb", "")
    
    # 提取所有存在的链
    chains = set()
    with open(pdb_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM"):
                chains.add(line[21:22])
                
    fixed_positions = {pdb_name: {}}
    
    # 一比一复刻 ProteinMPNN 的解析和补 X 逻辑
    for chain in chains:
        seq_dict = {}
        min_resn, max_resn = 1e6, -1e6
        
        with open(pdb_path, 'r') as f:
            for line in f:
                if line.startswith("ATOM") and line[21:22] == chain:
                    # 提取 22-26 位的编号和 27 位的插入码
                    resn_str = line[22:27].strip()
                    if resn_str[-1].isalpha(): 
                        resa, resn = resn_str[-1], int(resn_str[:-1])-1
                    else: 
                        resa, resn = "", int(resn_str)-1
                        
                    if resn < min_resn: min_resn = resn
                    if resn > max_resn: max_resn = resn
                    
                    if resn not in seq_dict: 
                        seq_dict[resn] = {}
                    if resa not in seq_dict[resn]: 
                        seq_dict[resn][resa] = True
        
        fixed_list = []
        seq_idx = 1 # ProteinMPNN 是从 1 开始的绝对索引
        
        # 从最小编号遍历到最大编号 (遇到断层会视为 X)
        for resn in range(int(min_resn), int(max_resn) + 1):
            if resn in seq_dict:
                for resa in sorted(seq_dict[resn]):
                    original_res_num = resn + 1
                    
                    # 检查是否处于 CDR-H3 设计区域
                    if chain == target_chain and design_start <= original_res_num <= design_end:
                        pass # 放开，让模型设计
                    else:
                        fixed_list.append(seq_idx) # 锁定框架
                    seq_idx += 1
            else:
                # 遇到断层，ProteinMPNN 会在这里塞一个 X。
                # 我们必须把这个 X 也锁定！否则序号就错位了。
                fixed_list.append(seq_idx)
                seq_idx += 1
                
        fixed_positions[pdb_name][chain] = fixed_list
        
    with open(output_file, 'w') as f:
        f.write(json.dumps(fixed_positions) + "\n")
        
    print(f"成功生成抗断层锁定配置: {output_file}")
    print(f"PDB Key: '{pdb_name}'")

if __name__ == "__main__":
    PDB_FILE = "/home/nvme03/yilin/ukantibody/inference/inference_antibody_5hi4_cdrh3/job_0_id_0_motif_5hi4_cdrh3_20260315_235116/job_0_id_0_motif_5hi4_cdrh3_20260315_235116.pdb" 
    
    create_fixed_positions(
        pdb_path=PDB_FILE, 
        target_chain="C",       # 👈 5hi4 的重链是 C 链
        design_start=105,       # 👈 紧接在框架 C104 之后的生成起始编号
        design_end=115,         # 👈 11个氨基酸的 CDR-H3 终止编号 (105 + 11 - 1 = 115)
        output_file="5hi4_fixed.jsonl" # 👈 输出文件名改为 5hi4
    )