import trimesh
import glob
import os

# meshes = []
# for obj in glob.glob("/workspace/root/miniconda3/nLOD/data/dz23/dz23-11/*.obj"):
#     meshes.append(trimesh.load(obj, process=False))
# #将所有纹理打包成一个纹理图集atlas,并调整 UV 坐标
# merged = trimesh.util.concatenate(meshes)
# merged.export("merged/dz23-11-merged.obj")

# 创建输出目录
os.makedirs("merged/dz23-11", exist_ok=True)

# 创建 Scene
scene = trimesh.Scene()

# 遍历加载每个 obj 并添加到 scene 中
for i, obj_path in enumerate(glob.glob("/workspace/root/miniconda3/nLOD/data/d1/DZ23_highRes/*.obj")):
    mesh = trimesh.load(obj_path, process=False)  # 保留纹理和材质3
    _ = mesh.vertex_normals
    mesh.fix_normals()
    # 可选：给每个几何体设置唯一的名称，便于在 OBJ 中区分 group/object
    geom_name = f"part_{i:04d}_{os.path.basename(obj_path)[:-4]}"
    scene.add_geometry(mesh, geom_name=geom_name)

# 导出为 OBJ（会自动导出 MTL 和所有独立的 PNG 纹理文件）
scene.export("merged/dz23-11/merged.obj", include_texture=True, include_normals=True)

