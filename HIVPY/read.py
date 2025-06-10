import os
from pathlib import Path

# 设置您的序列目录路径
sequence_dir = r"way\to\your\path"

# 获取所有FASTA文件
fasta_files = list(Path(sequence_dir).glob("*.fasta")) + \
              list(Path(sequence_dir).glob("*.fa")) + \
              list(Path(sequence_dir).glob("*.fna")) + \
              list(Path(sequence_dir).glob("*.txt"))

if not fasta_files:
    print("错误：目录中没有找到FASTA文件！")
    print(f"请检查路径: {sequence_dir}")
    print("支持的文件扩展名: .fasta, .fa, .fna, .txt")
    exit()

# 合并所有序列内容
combined_sequences = ""
for file_path in fasta_files:
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            # 确保每个序列之间有换行分隔
            if not content.endswith('\n'):
                content += '\n'
            combined_sequences += content
    except Exception as e:
        print(f"读取文件 {file_path.name} 时出错: {str(e)}")

# 添加Clustal Omega需要的格式说明
clustal_input = f"""# 以下是自动生成的 {len(fasta_files)} 个FASTA文件的合并内容
# 可直接粘贴到 Clustal Omega 在线版的输入框
# 总序列数: {combined_sequences.count('>')}
# 总字符数: {len(combined_sequences)}

{combined_sequences}"""

# 保存到桌面方便使用
desktop_path = Path.home() / "Desktop" / "input.fasta"
with open(desktop_path, 'w') as output_file:
    output_file.write(clustal_input)

print("="*80)
print("操作成功！")
print(f"包含文件数: {len(fasta_files)}")
print(f"总序列数: {combined_sequences.count('>')}")
print(f"输出文件位置: {desktop_path}")
