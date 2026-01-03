from ultralytics import YOLO


if __name__ == '__main__':
    # 直接使用预训练模型创建模型.
    # model.train(**{'cfg':'ultralytics/cfg/exp1.yaml', 'data':'dataset/data.yaml'})

    # 使用yaml配置文件来创建模型,并导入预训练权重.
    model = YOLO('E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8-ECAConv.yaml')
    result = model.train(data="E:\\SYQ\\yolov8\\ultralytics-main\\ceshi.yaml", epochs=200, model="E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8-ECAConv.yaml", imgsz=640, batch=16, workers=2)
    # model = YOLO('E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8-C2f-AFPN.yaml')
    # result = model.train(data="E:\\SYQ\\xiaorongshiyan2\\zhuyilijizhi\\yolov8-CBAM\\ultralytics-main\\GC-DET.yaml", epochs=200, model="E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8-C2f-AFPN.yaml", imgsz=640, batch=16, workers=2, resume=True)
    # model = YOLO('E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8_LCBHAM.yaml')
    # result = model.train(data="E:\\SYQ\\xiaorongshiyan2\\zhuyilijizhi\\yolov8-CBAM\\ultralytics-main\\GC-DET.yaml", epochs=200, model="E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\ultralytics\\cfg\\models\\v8\\yolov8_LCBHAM.yaml", imgsz=640, batch=16, workers=2)

# 模型验证
#     model = YOLO('E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\runs\\detect\\train195\\weights\\best.pt')
#     model = YOLO('E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\runs\\detect\\train200\\weights\\best.pt')
    # result = model.val(data="E:\\SYQ\\xiaorongshiyan2\\zhuyilijizhi\\yolov8-CBAM\\ultralytics-main\\GC-DET.yaml", epochs=200, model="E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\runs\\detect\\train195\\weights\\best.pt", imgsz=640, batch=16, workers=2)
    # result = model.val(data="E:\\SYQ\\yolov8\\ultralytics-main\\ceshi.yaml", epochs=200, model="E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\runs\\detect\\train200\\weights\\best.pt", imgsz=640, batch=16, workers=2)

    # 模型推理
    # model = YOLO('runs/detect/yolov8n_exp/best.pt')
    # model.predict(source='dataset/images/test', **{'save':True})

    # 模型导出
    # model.export(format="onnx")

