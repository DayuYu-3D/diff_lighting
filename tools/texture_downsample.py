from PIL import Image
import os

# 原地处理图片，直接覆盖旧文件
def resize_images_in_place(target_dir, max_resolution):
    # 确保目录存在
    if not os.path.exists(target_dir):
        print(f"错误：找不到目录 {target_dir}")
        return

    # 遍历指定目录中的所有文件
    for filename in os.listdir(target_dir):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            file_path = os.path.join(target_dir, filename)
            needs_save = False
            
            try:
                # 1. 打开图像文件
                with Image.open(file_path) as img:
                    # 检查分辨率是否超过指定大小
                    if img.size[0] > max_resolution[0] or img.size[1] > max_resolution[1]:
                        # 【核心安全操作】：使用 copy() 将图片数据加载到内存中
                        # 这样当 with 语句结束时，原文件会被安全关闭，解除系统占用
                        new_img = img.copy()
                        new_img.thumbnail(max_resolution)
                        needs_save = True
                        
                # 2. with 块结束后，原文件已经关闭，可以安全覆盖了
                if needs_save:
                    # 直接保存到原路径，原文件会被直接替换（相当于删除旧的存入新的）
                    new_img.save(file_path)
                    print(f"已缩小并覆盖: {filename} -> {max_resolution}")
                else:
                    # 如果尺寸合格，直接跳过，什么都不用做
                    print(f"跳过 (尺寸合格): {filename}")
                    
            except Exception as e:
                print(f"处理 {filename} 时出错: {e}")

if __name__ == "__main__":
    # 只需要指定你要处理的文件夹即可
    target_directory = "/workspace/root/miniconda3/nLOD/merged/dz23_2x1/Tile_X2_Y1"

    # 指定最大分辨率 (必须是2的次幂，如 512, 1024, 2048)
    max_resolution = (512, 512)

    # 调用函数
    print(f"开始处理目录: {target_directory}")
    resize_images_in_place(target_directory, max_resolution)
    print("处理完成！")