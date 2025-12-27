'''将pt转化为onnx
    onnx = 1.16.0
    opset = 12
'''
from ultralytics import YOLO


class Yolov8Pt2Onnx:
    def __init__(self, pt_path) -> None:
        self.convert(pt_path)

    def convert(self, pt_path):
        model = YOLO(pt_path)
        model.export(format="onnx", opset=12)


if __name__ == "__main__":
    pt_path = r"E:\SYQ\yolov8-recurrent\ultralytics-main\runs\detect\train43\weights\best.pt"
    Yolov8Pt2Onnx(pt_path)

