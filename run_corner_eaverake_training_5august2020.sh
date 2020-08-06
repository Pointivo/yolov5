export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64
export PATH=/usr/local/cuda-10.1/bin:$PATH
export CUDA_HOME=/usr/local/cuda-10.1
export CUDA_VISIBLE_DEVICES="1"
python train.py \
--cfg=models/yolov5x.yaml \
--data=corner_eave_rake.yaml \
--epochs=50 \
--batch-size=3 \
--img-size=640 \
--weights=weights/yolov5x.pt \
|& tee -a /home/pointivo/asim/yolov5/training.logs