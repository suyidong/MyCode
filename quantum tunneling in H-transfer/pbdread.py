from Bio.PDB import PDBParser
import numpy as np

# 创建PDB解析器
parser = PDBParser()

# 读取PDB文件
structure = parser.get_structure('2bbk', 'path/to/your/pbd_file')

# 定义两个原子之间的距离函数
def calculate_distance(atom1, atom2):
    """计算两个原子之间的欧几里得距离"""
    coord1 = atom1.get_coord()
    coord2 = atom2.get_coord()
    distance = np.linalg.norm(coord1 - coord2)
    return distance

# 遍历模型、链、残基、原子
for model in structure:
    for chain in model:
        for residue in chain:
            print(f"链: {chain.id}, 残基: {residue.id}")
            for atom in residue:
                print(f"原子: {atom.name}, 坐标: {atom.coord}")
                
            # 计算同一残基内两个原子之间的距离（如果有）
            if len(residue) >= 2:
                atom1 = residue.child_list[0]
                atom2 = residue.child_list[1]
                distance = calculate_distance(atom1, atom2)
                print(f"残基: {residue.id}, {atom1.name} 到 {atom2.name} 的距离: {distance:.2f} Å")
                
    # 计算不同链中两个原子之间的距离（示例）
    # 获取第一条链的第一个残基的第一个原子
    if len(model) > 0 and len(model[0]) > 0 and len(model[0][0]) > 0:
        atom_a = model[0][0][0].child_list[0]
    else:
        atom_a = None
    
    # 获取第二条链的第一个残基的第一个原子（如果存在）
    if len(model) > 0 and len(model[0]) > 1 and len(model[0][1]) > 0:
        atom_b = model[0][1][0].child_list[0]
    else:
        atom_b = None
    
    if atom_a and atom_b:
        distance_ab = calculate_distance(atom_a, atom_b)
        print(f"链 {model[0][0].id} 的 {atom_a.name} 到 链 {model[0][1].id} 的 {atom_b.name} 的距离: {distance_ab:.2f} Å")
