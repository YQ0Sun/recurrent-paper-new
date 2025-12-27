from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
if __name__ == '__main__':
    model = YOLO('E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\runs\\detect\\train\\weights\\best.pt') # loading pretrain weights
    results = model.predict(
                  model="E:\\SYQ\\yolov8-recurrent\\ultralytics-main\\runs\\detect\\train\\weights\\best.pt",
                  project='neu-experiments',
                  name='predict',
                  conf=0.3,
                  # source="E:\\SYQ\\yolov8-recurrent\\data\\PCB_DET2\\images\\test",
                  # source="E:\\SYQ\\xiaorongshiyan2\\zhuyilijizhi\\yolov8-CBAM\\data\\GC-DET\\images\\test",
                  source="E:\\SYQ\\yolov8\\data\\images\\test",
                  save=True
                )
