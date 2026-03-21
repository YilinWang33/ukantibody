import json

# 1. 这里的 target_name 必须和你传入 ProteinMPNN 的 pdb 文件名/jsonl中的 name 完全一致！
# 比如你输入的是 5mes.pdb，这里就是 "5mes"
target_name = "5mes" 
chain_id = "H"       # 假设你设计的 CDRH3 在 H 链上

bias_by_res_dict = {target_name: {chain_id: {}}}

# 2. 设定中间 Tip 区域 (你表格里 pLDDT 狂掉的区域，比如 101 到 106)
# 给柔性氨基酸加分，给体积大的芳香族氨基酸扣分
tip_positions = [101, 102, 103, 104, 105, 106]
for pos in tip_positions:
    bias_by_res_dict[target_name][chain_id][str(pos)] = {
        "G": 1.5, "S": 1.5, "D": 1.0, "N": 1.0, "P": 1.0,
        "Y": -1.5, "W": -1.5, "F": -1.5 
    }

# 3. 设定两端 Anchor 区域 (比如 97-100 和 107-108) 
# 保持一定的刚性和相互作用，可以偏好 Y, R, D
anchor_positions = [97, 98, 99, 100, 107, 108]
for pos in anchor_positions:
    bias_by_res_dict[target_name][chain_id][str(pos)] = {
        "Y": 1.5, "W": 1.0, "R": 1.0, "D": 1.0,
        "G": -1.0, "P": -1.0 
    }

# 4. 导出为 ProteinMPNN 能够识别的 jsonl 格式
with open("positional_bias.jsonl", "w") as f:
    f.write(json.dumps(bias_by_res_dict) + "\n")

print(f"为 {target_name} 生成的 Positional bias 已保存到 positional_bias.jsonl!")