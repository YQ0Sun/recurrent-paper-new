from PIL import Image, ImageDraw, ImageFont
import os
import xml.etree.ElementTree as ET

def draw_bbox_on_image(image_path, xml_path, save_folder):
    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 创建绘图对象
    draw = ImageDraw.Draw(image)

    # 解析 XML 文件
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 遍历每个目标框
    for obj in root.findall('object'):
        # 获取目标框的类别和边界框坐标
        name = obj.find('name').text
        xmin = int(obj.find('bndbox/xmin').text)
        ymin = int(obj.find('bndbox/ymin').text)
        xmax = int(obj.find('bndbox/xmax').text)
        ymax = int(obj.find('bndbox/ymax').text)

        # 绘制目标框矩形
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline='red', width=3)

        # 在目标框上方绘制类别名，调整字体大小
        font_size = 20  # 修改字体大小
        font = ImageFont.truetype("arial.ttf", font_size)  # 使用默认字体Arial或替换为您的字体文件
        text_width, text_height = draw.textsize(name, font=font)
        # if name == '1_chongkong':
        #     name = 'punching_hole'
        # elif name == '2_hanfeng':
        #     name = 'welding_line'
        # elif name == '3_yueyawan':
        #     name = 'crescent_gap'
        # elif name == '4_shuiban':
        #     name = 'water_spot'
        # elif name == '5_youban':
        #     name = 'oil_spot'
        # elif name == '6_siban':
        #     name = 'silk_spot'
        # elif name == '7_yiwu':
        #     name = 'inclusion'
        # elif name == '8_yahen':
        #     name = 'rolled_pit'
        # elif name == '9_zhehen':
        #     name = 'crease'
        # elif name == '10_yaozhed':
        #     name = 'waist folding'
        draw.text((xmin, ymin-25), name, fill='blue', font=font)

    # 保存绘制后的图像
    output_path = os.path.join(save_folder, os.path.basename(image_path))
    image.save(output_path)

# 文件夹路径
images_folder = r"E:\SYQ\yolov8\ultralytics-main\NEU-DET\IMAGES\crazing_1.jpg"
xmls_folder = r"E:\SYQ\yolov8\ultralytics-main\NEU-DET\ANNOTATIONS\crazing_1.xml"
save_folder = r"E:\SYQ\yolov8\ultralytics-main\NEU-DET\标签"

# 创建保存文件夹
if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# 遍历文件夹中的图片和对应的XML标签
for image_file in os.listdir(images_folder):
    if image_file.endswith(".jpg"):
        image_path = os.path.join(images_folder, image_file)
        xml_file = image_file.replace(".jpg", ".xml")
        xml_path = os.path.join(xmls_folder, xml_file)
        draw_bbox_on_image(image_path, xml_path, save_folder)


