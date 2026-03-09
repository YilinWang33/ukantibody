import os
from openfold.np import protein
from openfold.np.relax import relax

def relax_pdb(input_pdb, output_pdb):
    print(f"🚀 正在启动超算级 Amber 力场松弛: {input_pdb}")
    
    with open(input_pdb, "r") as f:
        prot = protein.from_pdb_string(f.read())
    
    # 补充了缺少的 use_gpu=True 参数
    amber_relaxer = relax.AmberRelaxation(
        max_iterations=0,      
        tolerance=2.39,        
        stiffness=10.0,        
        exclude_residues=[], 
        max_outer_iterations=20,
        use_gpu=False   # <====== 加上这个参数
    )
    
    print("⏳ 正在通过 GPU 梯度下降修复骨架断层与原子碰撞，请稍候...")
    relaxed_pdb_str, _, _ = amber_relaxer.process(prot=prot)
    
    with open(output_pdb, "w") as f:
        f.write(relaxed_pdb_str)
        
    print(f"✅ 修复完成！完美的抗体骨架已保存至: {output_pdb}")

if __name__ == "__main__":
    # 输入你刚才生成的 PDB
    INPUT_PDB = "inference/inference_antibody_5mes_cdrh3/job_0_id_0_motif_5mes_cdrh3_20260307_071134/job_0_id_0_motif_5mes_cdrh3_20260307_071134.pdb"
    # 输出的松弛后 PDB
    OUTPUT_PDB = "inference/inference_antibody_5mes_cdrh3/job_0_id_0_motif_5mes_cdrh3_20260307_071134/relaxed_5mes.pdb"
    
    relax_pdb(INPUT_PDB, OUTPUT_PDB)