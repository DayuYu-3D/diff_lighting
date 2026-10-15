import os
import trimesh
import numpy as np
from pathlib import Path
from PIL import Image
import PIL.Image

# 解除 Pillow 最大像素限制（支持 16K 贴图）
PIL.Image.MAX_IMAGE_PIXELS = None

MAX_TEXTURE_SIZE = 8192  # 最大贴图尺寸

def clean_and_export_textured_objs(input_dir, output_base_dir,
                                   white_threshold=220, saturation_threshold=0.2):
    in_path = Path(input_dir)
    out_path = Path(output_base_dir)
    
    obj_files = list(in_path.rglob('*.obj'))
    if not obj_files:
        print(f"在 {input_dir} 中未找到任何 obj 文件！")
        return

    print(f"共找到 {len(obj_files)} 个 OBJ 文件，开始逐个处理...")

    for obj_file in obj_files:
        print(f"\n--> 正在处理: {obj_file.name}")
        try:
            mesh = trimesh.load(str(obj_file), force='mesh', process=False, maintain_order=True)

            if not hasattr(mesh.visual, 'uv') or mesh.visual.uv is None:
                print(f"    跳过: {obj_file.name} 没有检测到 UV 纹理")
                continue

            # ================= 提取颜色 =================
            original_visual = mesh.visual
            mesh.visual = original_visual.to_color()
            face_colors = mesh.visual.face_colors
            mesh.visual = original_visual

            if face_colors is None or len(face_colors) == 0:
                # 所有面片都没有颜色，默认标记为空（删除）
                face_colors = np.zeros((len(mesh.faces), 4), dtype=np.uint8)

            # ================= 删除 UV=(0,1) 面片 =================
            faces_uv = mesh.visual.uv[mesh.faces]  # shape (n_faces, 3, 2)
            is_uv_zero_one = np.any(np.all(np.isclose(faces_uv, [0.0, 1.0]), axis=2), axis=1)

            # ================= 删除白色/透明/空颜色面片 =================
            r, g, b, a = face_colors[:,0], face_colors[:,1], face_colors[:,2], face_colors[:,3]

            # 透明面片
            is_transparent = (a == 0)
            # 空颜色
            is_empty_color = (r==0) & (g==0) & (b==0)
            # 白色面片
            Y = 0.299*r + 0.587*g + 0.114*b
            max_c = np.max(face_colors[:, :3], axis=1)
            min_c = np.min(face_colors[:, :3], axis=1)
            S = np.where(max_c == 0, 0, (max_c - min_c)/max_c)
            is_white_color = (Y >= white_threshold) & (S <= saturation_threshold)

            is_white_face = is_transparent | is_empty_color | is_white_color

            # ================= 组合删除条件 =================
            delete_mask = is_uv_zero_one | is_white_face
            keep_mask = ~delete_mask
            deleted_count = np.sum(delete_mask)

            if deleted_count > 0:
                keep_indices = np.where(keep_mask)[0]
                if len(keep_indices) == 0:
                    mesh.faces = []
                else:
                    # 安全保留子网格，保留 UV 映射
                    submeshes = mesh.submesh([keep_indices])
                    if len(submeshes) > 0:
                        mesh = submeshes[0]
                print(f"    成功删除 {deleted_count} 个面片 (UV=(0,1)/白色/透明/空颜色)")
            else:
                print("    没有找到需要删除的面片")

            if len(mesh.faces) == 0:
                print(f"    放弃导出: {obj_file.name} 所有面被删光了")
                continue

            # ================= 缩放大贴图 =================
            if hasattr(mesh.visual, 'material') and hasattr(mesh.visual.material, 'image') \
               and mesh.visual.material.image is not None:
                img = mesh.visual.material.image
                w, h = img.size
                if max(w, h) > MAX_TEXTURE_SIZE:
                    scale = MAX_TEXTURE_SIZE / max(w, h)
                    new_size = (int(w*scale), int(h*scale))
                    print(f"    缩放贴图: {w}x{h} -> {new_size[0]}x{new_size[1]}")
                    try:
                        mesh.visual.material.image = img.resize(new_size, Image.Resampling.LANCZOS)
                    except AttributeError:
                        mesh.visual.material.image = img.resize(new_size, Image.LANCZOS)

            # ================= 导出 =================
            rel_path = obj_file.relative_to(in_path)
            output_file_path = out_path / rel_path
            output_file_path.parent.mkdir(parents=True, exist_ok=True)
            if output_file_path.suffix.lower() != '.obj':
                output_file_path = output_file_path.with_suffix('.obj')

            mesh.export(str(output_file_path))
            print(f"    已保存: {output_file_path}")

        except Exception as e:
            print(f"    [错误] 处理 {obj_file.name} 时发生异常: {e}")

    print("\n=================================")
    print("全部批处理任务完成！")


# ================= 示例使用 =================
if __name__ == "__main__":
    INPUT_DIRECTORY = r"/workspace/root/miniconda3/nLOD/results_d1/0.05/meshes"
    OUTPUT_DIRECTORY = r"/workspace/root/miniconda3/nLOD/results_d1/0.05/meshes_noWhite"

    clean_and_export_textured_objs(INPUT_DIRECTORY, OUTPUT_DIRECTORY)