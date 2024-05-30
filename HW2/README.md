# 框架使用
Faster-RCNN和Yolov3的训练都采用了mmdetection框架。，可以使用如下命令克隆其项目：
```
git clone https://github.com/open-mmlab/mmdetection.git
```
# 数据集准备
首先下载Voc2007数据集，可使用如下命令下载：
```
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar

# 数据格式如下，需将其解压到mmdetection目录下的data文件夹中
# ~/
# └── VOCdevkit/
#     ├── VOC2007/
#     │   ├── Annotations/
#     │   ├── ImageSets/
#     │   ├── JPEGImages/
#     │   ├── SegmentationClass/
#     │   └── SegmentationObject/
```
# Faster-RCNN 训练
1. 将本项目里的config/faster-rcnn/faster-rcnn_50_fpn_1x_voc.py放入mmdetection的configs/faster-rcnn文件夹中
2. 将mmdetection/configs/\_\_base\_\_/models/faster-rcnn_r50_fpn.py替换成config/faster-rcnn/faster-rcnn_r50_fpn.py
3. 将mmdetection/configs/\_\_base\_\_/datasets/s/voc0712.py替换成config/faster-rcnn/voc0712.py
4. 将mmdetection/configs/\_\_base\_\_/schedules/schedule_1x.py替换成config/faster-rcnn/schedule_1x.py
5. 将mmdetection/configs/\_\_base\_\_/default_runtime.py 替换成config/faster-rcnn/default_runtime.py

6. 最后在mmdetection目录下运行
```
python tools/train.py config/faster-rcnn/faster-rcnn_50_fpn_1x_voc.py
``` 

# Yolov3训练
1. 将mmdetection/configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py替换成本项目里的 config/yolov3/yolov3_d53_8xb8-ms-608-273e_coco.py

2. 最后运行
```
python tools/train.py configs/yolo/yolov3_d53_8xb8-ms-608-273e_coco.py
```

# Stable Diffusion 图片生成
本任务第三阶段要求生成不在测试集中的包含测试集中类别的三张图片，笔者第一时间想到了生成模型， 故利用生成模型生成了这三张图片，最后运行如下代码即可
```
python txt2img.py --label 'class_name' --checkpoint ckpts/stable-diffusion-2-1-base
```

# 模型推理
运行以下代码即可，img_dir是想要检测的图片所在目录， out_dir是输出地址，config是配置文件，如config/faster-rcnn/faster-rcnn_50_fpn_1x_voc.py， checkpoint 是保存的权重
```
python inference.py --img_dir your_img_dir --out_dir out_dir --config config --checkpoint model_checkpoint.pth
```
