python train.py \
--cfg=models/yolov5x.yaml \
--data=penetration.yaml \
--epochs=50 \
--batch-size=3 \
--img-size=1024 \
--noautoanchor \
--weights=weights/yolov5x.pt \
|& tee -a /home/ubuntu/yolov5/yolov5/training.logs