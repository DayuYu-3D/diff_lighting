import trimesh
import glob
import os
import re
from collections import defaultdict

#### 运行完后，运行texture-donwsaple.py将纹理降采样到512

# 1. 路径配置
INPUT_DIR = "/workspace/root/miniconda3/nLOD/data/d1/DZ23_highRes"
# 更改了输出文件夹名称，以反映 2x1 的合并方式
OUTPUT_DIR = "merged/dz23_2x1" 
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 用于存放分组的字典，格式: {(新X, 新Y): [文件路径1, 文件路径2...]}
grid_groups = defaultdict(list)

print("正在扫描文件并进行 2x1 (水平两两合并) 分组...")

# 专门匹配 "Tile_+005_+003..." 格式的正则表达式
pattern = re.compile(r"Tile_([+-]?\d+)_([+-]?\d+)")

# 2. 遍历并分组
for obj_path in glob.glob(os.path.join(INPUT_DIR, "*.obj")):
    filename = os.path.basename(obj_path)
    
    match = pattern.search(filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        
        # === 核心修改部分 ===
        # 将 X 轴两两合并 (0和1变成0, 2和3变成1...)
        new_x = x // 2 
        # Y 轴保持原样不变，这样就是水平方向相邻的 2 个网格合并
        new_y = y      
        
        grid_groups[(new_x, new_y)].append(obj_path)
    else:
        print(f"警告：无法从文件名 {filename} 中解析出坐标，已跳过。")

print(f"分组完成！共划分为 {len(grid_groups)} 个合并大网格。开始合并...")

# 3. 按组进行合并与导出
for (gx, gy), file_paths in grid_groups.items():
    print(f"正在处理合并网格 Grid_{gx}_{gy} (包含 {len(file_paths)} 个子文件)...")
    
    scene = trimesh.Scene()
    
    for i, obj_path in enumerate(file_paths):
        try:
            # 加载网格，保留原有纹理材质
            mesh = trimesh.load(obj_path, process=False)
            _ = mesh.vertex_normals
            mesh.fix_normals()
            
            # 设置唯一名称避免冲突
            geom_name = f"part_{i:02d}_{os.path.basename(obj_path)[:-4]}"
            scene.add_geometry(mesh, geom_name=geom_name)
            
        except Exception as e:
            print(f"  读取文件 {os.path.basename(obj_path)} 失败: {e}")
            
    # 如果该组内成功加载了模型，则导出到专属子文件夹
    if not scene.is_empty:
        # 1. 定义子文件夹的名称 (例如: Tile_X2_Y1)
        subfolder_name = f"Tile_X{gx}_Y{gy}"
        # 2. 拼接出子文件夹的完整路径
        subfolder_path = os.path.join(OUTPUT_DIR, subfolder_name)
        # 3. 创建子文件夹 (exist_ok=True 避免重复运行时报错)
        os.makedirs(subfolder_path, exist_ok=True)
        
        # 4. 定义导出的 OBJ 文件名及完整路径 (保存在子文件夹内)
        out_filename = f"merged_{subfolder_name}.obj"
        out_filepath = os.path.join(subfolder_path, out_filename)
        
        # 导出为 OBJ，包含独立的 PNG 纹理和 MTL
        scene.export(out_filepath, include_texture=True, include_normals=True)
        print(f"  -> 已导出至子文件夹: {subfolder_name}/{out_filename}")

print("所有网格合并完毕！")