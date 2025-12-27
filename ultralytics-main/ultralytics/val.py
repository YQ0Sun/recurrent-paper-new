from ultralytics import YOLO

if __name__ == '__main__':
    # 直接使用预训练模型创建模型.
    # model = YOLO('yolov8n.pt')
    # model.train(**{'cfg':'ultralytics/cfg/exp1.yaml', 'data':'dataset/data.yaml'})

    # 使用yaml配置文件来创建模型,并导入预训练权重.
    model = YOLO('E:\\SYQ\\yolov8\\ultralytics-main\\runs\\detect\\train3\\weights\\best.pt')
    result = model.val(data="E:\\SYQ\\yolov8\\ultralytics-main\\ceshi.yaml", epochs=200, model="E:\\SYQ\\yolov8\\ultralytics-main\\runs\\detect\\train3\\weights\\best.pt", imgsz=640, batch=16, workers=2)
    # 模型验证
    # model = YOLO('runs/detect/train11/weights/best.pt')
    # model.val(**{'cfg':'ultralytics/cfg/PCB.yaml', 'data':'datasets/VOCPCB.yaml'})

    # 模型推理
    # model = YOLO('runs/detect/yolov8n_exp/best.pt')
    # model.predict(source='dataset/images/test', **{'save':True})

    # 模型导出
    # model = YOLO("Weight/yolov8n.pt")  # load an official model
    # model.export(format="onnx")