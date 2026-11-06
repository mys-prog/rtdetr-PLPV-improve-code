from ultralytics import RTDETR
import warnings
warnings.filterwarnings('ignore')

model = RTDETR(r"ultralytics/cfg/models/rt-detr/rtdetr-l-GPA-CABM.yaml")
# model.load('rtdetr-l.pt') # 是否加载预训练权重
model.train(data='mydata.yaml',  # 训练参数均可以重新设置
                        epochs=150,
                        imgsz=320,
                        workers=0,
                        batch=4,
                        device=0,
                        optimizer='AdamW',
                        close_mosaic=60,
                        cos_lr=False,
                        iou=0.3,
                        lr0=1E-3,
                        patience=100,
                        )