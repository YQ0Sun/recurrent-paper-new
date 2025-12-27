import os
import shutil

# 定义图片名称
target_images = [
    # C2f模块
    # "crazing_197.jpg",
    # "inclusion_137.jpg",
    # "patches_23.jpg",
    # "patches_173.jpg",
    # "pitted_surface_4.jpg",
    # "rolled-in_scale_165.jpg",
    # "rolled-in_scale_221.jpg",
    # "scratches_206.jpg",

    "crazing_81.jpg",
    "inclusion_137.jpg",
    "patches_115.jpg",
    "pitted_surface_61.jpg",
    "rolled-in_scale_32.jpg",
    "scratches_161.jpg"

    # "04_mouse_bite_11.jpg",
    # "04_spurious_copper_10.jpg",
    # "04_spurious_copper_15.jpg",
    # "11_mouse_bite_02.jpg"
    # 预测
    # "img_02_436151900_00101.jpg",
    # "img_02_436153600_00665.jpg",
    # "img_02_436153600_00691.jpg",
    # "img_02_436153600_00695.jpg",
    # "img_03_424992300_00311.jpg",
    # "img_03_425501800_01178.jpg",
    # "img_04_4406742400_00645.jpg",
    # "img_07_435974600_00214.jpg",
    # "img_08_3403334300_00853.jpg",
    # "img_08_4406743300_00405.jpg"
]

# 遍历的文件夹路径
source_folder = 'E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\neu-experiments\\predict42'  # 修改为你实际的文件夹路径

# 目标文件夹
destination_folder = os.path.join(os.path.dirname(source_folder), "predict42-NEU")

# 如果目标文件夹不存在，创建它
# if not os.path.exists(destination_folder):
os.makedirs(destination_folder, exist_ok=True)

# 遍历文件夹，查找符合条件的图片
for root, dirs, files in os.walk(source_folder):
    for file in files:
        if file in target_images:
            # 将符合条件的图片复制到目标文件夹
            source_path = os.path.join(root, file)
            destination_path = os.path.join(destination_folder, file)
            shutil.copy2(source_path, destination_path)
            print(f"已复制: {file}")

print("所有符合条件的图片已复制完成。")
