import biotite.structure.io as strucio

# 替换成你实际报错的 PDB 路径
pdb_path = "/DATA/disk3/yilin/dyMEAN/all_data/pdb/4ffv.pdb"
array = strucio.load_structure(pdb_path, model=1)

def get_actual_blocks(chain_id, start, end):
    # 筛选指定链、非杂原子、在起始范围内的原子
    mask = (array.chain_id == chain_id) & (array.hetero == False) & (array.res_id >= start) & (array.res_id <= end)
    atoms = array[mask]
    
    # 提取实际存在的连续残基段 (完全模拟 La-proteina 的提取逻辑)
    seen = set()
    res_ids = []
    for r in atoms.res_id:
        if r not in seen:
            seen.add(r)
            res_ids.append(r)
            
    if not res_ids: 
        return "", 0
        
    blocks = []
    block_start = res_ids[0]
    prev = res_ids[0]
    total_residues = 0
    
    for r in res_ids[1:]:
        if r != prev + 1:
            blocks.append(f"{chain_id}{block_start}-{prev}")
            total_residues += (prev - block_start + 1)
            block_start = r
        prev = r
        
    blocks.append(f"{chain_id}{block_start}-{prev}")
    total_residues += (prev - block_start + 1)
    
    return "/".join(blocks), total_residues

# 你原始期望的区间
regions = [
    ("A", 1, 395),
    ("L", 1, 108),
    ("H", 1, 96)
]
region_after = ("H", 107, 119)
gen_length = 10  # 你想生成的 CDRH3 的长度

final_contig_parts = []
motif_total_length = 0

# 1. 遍历前半部分固定区
for chain, start, end in regions:
    blocks_str, length = get_actual_blocks(chain, start, end)
    if blocks_str:
        final_contig_parts.append(blocks_str)
        motif_total_length += length
        
# 2. 插入要生成的氨基酸长度
final_contig_parts.append(f"{gen_length}-{gen_length}")

# 3. 加上后半部分固定区
blocks_str, length = get_actual_blocks(region_after[0], region_after[1], region_after[2])
if blocks_str:
    final_contig_parts.append(blocks_str)
    motif_total_length += length

final_contig = "/".join(final_contig_parts)
print(f"✅ 修正后的 contig_string: {final_contig}")
print(f"✅ 修正后的 motif_min_length & motif_max_length: {motif_total_length + gen_length}")