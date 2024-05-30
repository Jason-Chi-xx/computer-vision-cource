# Copyright (c) OpenMMLab. All rights reserved.
import asyncio
import numpy as np
from argparse import ArgumentParser

from mmdet.apis import inference_detector, init_detector

import mmcv
from mmdet.registry import VISUALIZERS
import os

import warnings 
warnings.filterwarnings("ignore") 


def get_img_file(file_name):
    imagelist = []
    for parent, dirnames, filenames in os.walk(file_name):
        for filename in filenames:
            if filename.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                imagelist.append(os.path.join(parent, filename))
        
        return imagelist

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--img_dir', default='/home/xcm/CV_HW2/mmdetection/infer_imgs',help='Image file')
    parser.add_argument('--out_dir', default='./infer_output_yolov3/',help='Image file')
    parser.add_argument('--config',default='/home/xcm/CV_HW2/mmdetection/configs/faster_rcnn/faster-rcnn_50_fpn_1x_voc.py', help='Config file')
    parser.add_argument('--checkpoint',default='/home/xcm/CV_HW2/mmdetection/work_dirs/faster-rcnn_50_fpn_1x_voc/epoch_20.pth', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference') #cuda:0
    parser.add_argument(
        '--score-thr', type=float, default=0.9, help='bbox score threshold')
    # parser.add_argument(
    #     '--async-test',
    #     action='store_true',
    #     help='whether to set async options for async inference.')
    args = parser.parse_args()
    return args


def main(args):
    
    model = init_detector(args.config, args.checkpoint, device=args.device)

    out_dir = args.out_dir
    if not os.path.exists(out_dir):  #判断是否存在文件夹如果不存在则创建为文件夹
       os.makedirs(out_dir)

    images = get_img_file(args.img_dir)
    
    for image in images:
        print(image)
        # 测试单张图片并展示结果
        img = mmcv.imread(image)    # 或者 ，这样图片仅会被读一次 img = 'demo.jpg'
        result = inference_detector(model, img)
        
        #bboxes_scores = np.vstack(result)
        #bboxes=bboxes_scores[:,:4]
        #score=bboxes_scores[:,4]
        #labels = [
        #           np.full(bbox.shape[0], i, dtype=np.int32)
        #           for i, bbox in enumerate(result)
        #       ]
        #labels = np.concatenate(labels)
        #print(bboxes_scores)
        #print(labels)
        #print(result)
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
        out_file = out_dir + image.split('/')[-1]
        visualizer.add_datasample(
            'result',
            img,
            data_sample=result,
            draw_gt=False,
            wait_time=0,
            out_file=out_file # optionally, write to output file
        )
        # visualizer.show()
        # import pdb; pdb.set_trace()
        # out_file = out_dir + image.split('/')[-1]
        # model.show_result(img,result, score_thr=args.score_thr,out_file=out_file)
    




if __name__ == '__main__':
    args = parse_args()
    main(args)
