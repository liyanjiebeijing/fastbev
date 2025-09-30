
from PIL import Image
import os

def process_image(input_path, output_dir):
    """执行图像缩放、旋转、裁剪操作"""
    try:
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)

        # 打开原始图像
        original_img = Image.open(input_path)
        print(f"原始图像尺寸: {original_img.size}")

        # 1. 缩放图像 (缩小50%)
        scaled_img = original_img.resize(
            (int(original_img.width * 0.5), int(original_img.height * 0.5)),
            Image.Resampling.LANCZOS
        )
        scaled_path = os.path.join(output_dir, "scaled.jpg")
        scaled_img.save(scaled_path)
        print(f"缩放后保存至: {scaled_path}")

        # 2. 旋转图像 (逆时针45度)
        rotated_img = original_img.rotate(45, expand=True)
        rotated_path = os.path.join(output_dir, "rotated.jpg")
        rotated_img.save(rotated_path)
        print(f"旋转后保存至: {rotated_path}")

        # 3. 裁剪图像 (中心区域)
        left = original_img.width * (-0.25)
        top = original_img.height * (-0.3)
        right = original_img.width * 0.75
        bottom = original_img.height * 0.75
        cropped_img = original_img.crop((left, top, right, bottom))
        cropped_path = os.path.join(output_dir, "cropped.jpg")
        cropped_img.save(cropped_path)
        print(f"裁剪后保存至: {cropped_path}")

    except Exception as e:
        print(f"处理失败: {str(e)}")

if __name__ == "__main__":
    # 使用示例
    input_image = "./data/lena.jpg"  # 替换为你的图片路径
    output_folder = "output"
    process_image(input_image, output_folder)
