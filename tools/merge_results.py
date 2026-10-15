import trimesh
from pathlib import Path
import os

# 输入与输出路径
INPUT_ROOT = "/workspace/root/miniconda3/nLOD/data/d1/DZ23_merged"
OUTPUT_DIR = "/workspace/root/miniconda3/nLOD/results_d1/"
OUTPUT_OBJ = Path(OUTPUT_DIR) / "merged123.obj"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 创建 Scene
scene = trimesh.Scene()

# 查找所有 OBJ 文件
obj_files = list(Path(INPUT_ROOT).rglob("*.obj"))

if not obj_files:
    print(f"未找到任何 OBJ 文件在 {INPUT_ROOT}")
else:
    print(f"找到 {len(obj_files)} 个 OBJ 文件，开始处理...")

for i, obj_path in enumerate(obj_files):
    mesh = trimesh.load(str(obj_path), process=False)
    mesh.fix_normals()

    # 给每个几何体设置唯一名称
    geom_name = f"part_{i:04d}_{obj_path.stem}"

    # ===== 处理材质贴图重命名 =====
    visual = mesh.visual
    if hasattr(visual, "material"):
        
        # 【修复1：解决材质名冲突】 极其重要！必须给每个碎片赋予独立的材质名称
        unique_mat_name = f"mat_{geom_name}"
        visual.material.name = unique_mat_name
        
        if hasattr(visual.material, "image") and visual.material.image is not None:
            # 【修复2：保持 PIL Image 对象，切勿赋值为路径】
            img = visual.material.image 
            
            # 给内部 Image 对象打上文件名标签
            # 导出时 Trimesh 会自动使用这个名字，并将图片随 mtl 一起保存到磁盘
            img.filename = f"{unique_mat_name}_tex.png"
            visual.material.image = img

    # 将几何体加入 Scene
    scene.add_geometry(mesh, geom_name=geom_name)

# 导出 OBJ，会生成对应 MTL 并在同目录自动吐出所有独立命名的贴图
print("\n正在合并并导出 Scene，这可能需要一点时间...")
scene.export(str(OUTPUT_OBJ), include_texture=True, include_normals=True)

print(f"\n完成：已合并 {len(obj_files)} 个 OBJ 到 {OUTPUT_OBJ}")
print(f"对应的 MTL 和贴图已自动保存到 {OUTPUT_DIR}")