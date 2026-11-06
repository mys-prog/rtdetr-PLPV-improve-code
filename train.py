from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')

model = RTDETR(r"ultralytics/cfg/models/rt-detr/PL-DETR.yaml")
# model.load('rtdetr-l.pt')
model.train(data='mydata.yaml',
                        epochs=150,
                        imgsz=320,
                        workers=0,
                        batch=4,
                        device=0,
                        )