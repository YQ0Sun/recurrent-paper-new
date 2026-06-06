import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO



if __name__ == '__main__':
    # model = YOLO('E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\runs\\detect\\train261\\weights\\best.pt') # yolov8+LEF,多尺度
    model = YOLO('E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\runs\\detect\\train195\\weights\\best.pt') # yolov8+EMFF,多尺度
    model.predict(source='E:\\SYQ\\yolov8\\data\\images\\test',
                  imgsz=640,
                  project='runs/detect/feature',
                  name='yolov8',
                  save=True,
                  # conf=0.2,

                  # iou=0.7,
                  # agnostic_nms=True,
                  visualize=True, # visualize model features maps
                  # line_width=2, # line width of the bounding boxes
                  # show_conf=False, # do not show prediction confidence
                  # show_labels=False, # do not show prediction labels
                  # save_txt=True, # save results as .txt file
                  # save_crop=True, # save cropped images with results
                )