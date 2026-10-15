import os
import argparse
import trimesh
import shutil

def process_obj(input_path, output_dir_for_this_obj):
    """
    加载 OBJ 文件，确保计算顶点法线，然后导出到专用子文件夹。
    trimesh 会自动导出伴生的 .mtl 和纹理文件到同一目录，避免文件名冲突。
    """
    try:
        # 加载网格（force='mesh' 以确保是 Trimesh 对象）
        mesh = trimesh.load(input_path, force='mesh')
        
        # 触发法线计算（如果原始 OBJ 没有 vn，自动计算）
        _ = mesh.vertex_normals
        
        # 创建该 OBJ 专属的输出子文件夹（避免 MTL/纹理文件名冲突）
        os.makedirs(output_dir_for_this_obj, exist_ok=True)
        
        # 导出路径：子文件夹内保持原 OBJ 文件名
        output_path = os.path.join(output_dir_for_this_obj, os.path.basename(input_path))
        
        # 导出 OBJ（trimesh 会自动把 .mtl 和纹理文件导出到同一目录）
        mesh.export(output_path)
        
        print(f"成功处理并导出（已添加法线，已隔离 MTL/纹理）：{os.path.basename(input_path)} → {output_dir_for_this_obj}")
        
    except Exception as e:
        print(f"处理失败 {os.path.basename(input_path)}: {e}")

#python calOBJ_normals.py --input_dir /workspace/root/miniconda3/nLOD/data/dz23/merged --output_dir output_objs  [--recursive]
def main():
    parser = argparse.ArgumentParser(description="批量读取目录下所有 OBJ 文件，添加法线后导出到指定文件夹（每个 OBJ 独立子文件夹，避免 MTL/纹理覆盖）")
    parser.add_argument('--input_dir', type=str, required=True, help='输入目录（包含 OBJ 文件）')
    parser.add_argument('--output_dir', type=str, required=True, help='输出根目录（每个 OBJ 会创建独立子文件夹）')
    parser.add_argument('--recursive', action='store_true', help='是否递归遍历子目录（默认仅当前目录，会保留原相对路径结构）')
    
    args = parser.parse_args()
    
    # 创建输出根目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 遍历文件
    if args.recursive:
        for root, _, files in os.walk(args.input_dir):
            for file in files:
                if file.lower().endswith('.obj'):
                    input_path = os.path.join(root, file)
                    # 计算相对路径（保留原目录结构）
                    rel_path = os.path.relpath(input_path, args.input_dir)
                    rel_dir = os.path.dirname(rel_path)
                    base_name = os.path.splitext(file)[0]
                    
                    # 输出子文件夹：output_dir / (原相对目录) / (OBJ文件名去扩展)
                    output_subdir = os.path.join(args.output_dir, rel_dir, base_name) if rel_dir != '.' else os.path.join(args.output_dir, base_name)
                    
                    process_obj(input_path, output_subdir)
    else:
        for file in os.listdir(args.input_dir):
            if file.lower().endswith('.obj'):
                input_path = os.path.join(args.input_dir, file)
                base_name = os.path.splitext(file)[0]
                
                # 输出子文件夹：output_dir / (OBJ文件名去扩展)
                output_subdir = os.path.join(args.output_dir, base_name)
                
                process_obj(input_path, output_subdir)

if __name__ == '__main__':
    main()