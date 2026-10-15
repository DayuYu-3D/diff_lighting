import os
import sys
from pathlib import Path

def parse_obj_file(filepath):
    """
    解析单个 OBJ 文件，返回 (顶点数, 面元数)
    只统计以 'v ' 开头的顶点行（几何顶点）和 'f ' 开头的面行
    """
    vertices = 0
    faces = 0
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                if not parts:
                    continue
                prefix = parts[0]
                if prefix == 'v' and len(parts) >= 4:  # v x y z
                    vertices += 1
                elif prefix == 'f':
                    # 统计面上的顶点索引数量，只要 >=3 就算一个有效面
                    face_vertices = len(parts) - 1
                    if face_vertices >= 3:
                        faces += 1
    except Exception as e:
        print(f"错误读取文件 {filepath}: {e}")
        return 0, 0
    return vertices, faces

def main(folder_path):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        print(f"文件夹不存在: {folder_path}")
        return
    
    obj_files = list(folder.rglob("*.obj"))  # 支持子文件夹，改成 glob("*.obj") 只当前目录
    if not obj_files:
        print("文件夹中未找到 .obj 文件")
        return
    
    total_vertices = 0
    total_faces = 0
    
    print(f"找到 {len(obj_files)} 个 OBJ 文件，开始统计...\n")
    
    for i, file_path in enumerate(obj_files, 1):
        v, f = parse_obj_file(file_path)
        total_vertices += v
        total_faces += f
        print(f"[{i}/{len(obj_files)}] {file_path.name}: 顶点 {v:,} | 面 {f:,}")
    
    print("\n" + "="*50)
    print(f"所有文件总计:")
    print(f"总顶点数: {total_vertices:,}")
    print(f"总面元数: {total_faces:,}")
    print(f"总文件数: {len(obj_files)}")

if __name__ == "__main__":
    
    folder_path = '/workspace/root/miniconda3/nLOD/data/dz23/DZ23_highRes'
    main(folder_path)