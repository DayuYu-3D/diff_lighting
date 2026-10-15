import os
from pathlib import Path
from PIL import Image

def analyze_image_directory(directory_path):
    """
    统计指定目录下 JPG 和 PNG 文件的数量及总像素。
    默认会递归搜索所有子目录。
    """
    # 初始化统计数据
    stats = {
        'jpg_count': 0,
        'png_count': 0,
        'total_pixels': 0,
        'error_count': 0
    }
    
    target_dir = Path(directory_path)
    
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"错误: 目录 '{directory_path}' 不存在或不是一个有效的文件夹。")
        return stats

    print(f"正在扫描目录: {target_dir.absolute()} ...\n")

    # 使用 rglob 进行递归查找，如果你只需要查当前文件夹，可以改为 target_dir.iterdir()
    for file_path in target_dir.rglob('*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()
            
            # 判断文件类型
            is_jpg = ext in ['.jpg', '.jpeg']
            is_png = ext == '.png'
            
            if not (is_jpg or is_png):
                continue
                
            # 记录文件数量
            if is_jpg:
                stats['jpg_count'] += 1
            elif is_png:
                stats['png_count'] += 1
                
            # 获取像素数量
            try:
                # Image.open 默认采用懒加载机制，只读取文件头获取 size 属性，非常快速
                with Image.open(file_path) as img:
                    width, height = img.size
                    stats['total_pixels'] += (width * height)
            except Exception as e:
                stats['error_count'] += 1
                print(f"警告: 无法读取图片 {file_path.name} (可能已损坏) - {e}")

    return stats

if __name__ == "__main__":
    # 请在这里替换为你需要统计的目录路径
    # 可以是相对路径 './images' 或绝对路径 'D:/data/images'
    folder_to_scan = "/workspace/root/miniconda3/nLOD/data/d1/DZ23_highRes" 
    
    results = analyze_image_directory(folder_to_scan)
    
    # 格式化输出结果
    print("-" * 30)
    print("统计完成！结果如下：")
    print(f"JPG/JPEG 文件数: {results['jpg_count']} 个")
    print(f"PNG 文件数:      {results['png_count']} 个")
    print(f"总计图片数量:    {results['jpg_count'] + results['png_count']} 个")
    # 使用格式化字符串添加千位分隔符，方便读取大数字
    print(f"总计像素数量:    {results['total_pixels']:,} 像素")
    
    if results['error_count'] > 0:
        print(f"\n注意: 有 {results['error_count']} 个文件在读取时发生错误。")