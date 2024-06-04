
'''
    代码：使用from multiprocessing import Event, Process,Manager,Queue 来进行并行预测
    输入：检测数据集，直接在参数里的source=ROOT / 'final_test_data/images2'改
    输出：使用了人脸、电动车的检测模型，将预测结果输出到ROOT\runs\multi_predict中
    这两个模型的预测使用了不同的GPU，通过115行的device="0"/"1"来更换设备序号
'''

##预测用208行，打标签用209行
import multiprocessing
import time
import argparse
import os
import platform
import sys
from itertools import count
from pathlib import Path
import numpy as np
import time
import json
import torch




FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory yolov5项目路径
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH    ； 将detect所在路径存入 path下，否则from将无法导入同路径下的文件
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative



from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode


def run( #分成6个部分
        source=ROOT / 'final_test_data/images6_small',  # file/dir/URL/glob/screen/0(webcam)
        #source = "E:\\project_code\\instance_fenge\\mask_rcnn\\img_mohu7.jpg",
        data=ROOT / 'final_test_data/widerface.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.5,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        save_crop=False,  # save cropped prediction boxes
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        line_thickness=1,  # bounding box thickness (pixels)

        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride

):
    #print(class_num)
    #####################！！！！！！！！！！单类检测与融合 所需的参数！！！！！！！！！！！#################
    ##1. labels的txt保存
    save_txt = True
    ##2. 循环检测所使用的权重链表
    weight0 = ROOT / 'single_class_weight/face_model.pt'
    weight1 = ROOT / 'single_class_weight/person_model.pt'
    weight2 = ROOT / 'single_class_weight/motorcar_model.pt'
    weight_list = [weight0, weight1, weight2]
    ##3. 目标类型
    class_number = [0, 1, 2] #0:人脸； 1：人 ； 2：电动车。
    ##4. 图片类型 --- 暂时没用到~~~
    image_class = ["motorcar", "person", "face", "motorcar_person", "motorcar_face", "person_face",
                   "motorcar_person_face"]
    #####################！！！！！！！！！！单类检测与融合 所需的参数！！！！！！！！！！！#################



    ####################！！！！！！固定的！！！！！！####################！！！！！！！！！！！！！！！！！！
    ################################### source的传入与判断
    source = str(source) #测试数据集的路径
    #save_img = not nosave and not source.endswith('.txt')  # save inference images

    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS) #判断是否是正确的图片/视频格式
    # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')) #判断是不是url格式
    # webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    # screenshot = source.lower().startswith('screen')
    # if is_url and is_file:
    #     source = check_file(source)  # download
    ################################### 新建保存结果的文件夹
    # Directories #预测结果的保存路径 run/detect/exp15
    #save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    save_dir = ROOT / "runs/multi_predict"
    # 创建exp15文件夹 与 对应的labels文件夹
    #(save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    ####################！！！！！！固定的！！！！！！####################！！！！！！！！！！！！！！！！！！！


    ###较大更改---1：权重模型的更换
    weights = weight_list[2]
    ###较大更改---2：类别序号的更换，三类检测分别是人脸、人、电动车
    class_code = class_number[2] #对应205行的class更改

    device = "cpu"
    device = select_device(device)
    print(device)  #cuda:0

    model2 = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model2.stride, model2.names, model2.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size


    # Dataloader 加载待预测的图片---没改
    bs = 1  # batch_size
    # if webcam:
    #     view_img = check_imshow()
    #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    #     bs = len(dataset)
    # elif screenshot:
    #     dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    # else:
    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference 模型运行---没改
    model2.warmup(imgsz=(1 if pt or model2.triton else bs, 3, *imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

    result = {}
    for path, im, im0s, vid_cap, s in dataset:#---没改
        with dt[0]:
            im = torch.from_numpy(im).to(model2.device)
            im = im.half() if model2.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model2(im, augment=augment, visualize=visualize)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions

        for i, det in enumerate(pred):  # per image
            ########！！！！！！！！！！一轮循环就是一张图片！！！！！#########
            seen += 1
            # if webcam:  # batch_size >= 1
            #     p, im0, frame = path[i], im0s[i].copy(), dataset.count
            #     s += f'{i}: '
            # else:
            p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            img_name = p.name[:-4]
            save_path = str(save_dir / p.name)  # im.jpg 生成保存图片的路径：exp17/1.png 这种
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # im.txt 生成保存txt文档的路径：exp1/labels/1.png这样
            s += '%gx%g ' % im.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            imc = im0.copy() if save_crop else im0  # for save_crop
            annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                boxes = []
                labels = []
                scores = []
                for *xyxy, conf, cls in reversed(det):
                    ######！！！！！！！！！！生成单个图片中的 单个目标的相对坐标！！！！！！！！！###########
                    some_xyxy = xyxy
                    boxes.append(some_xyxy)
                    cls = 4
                    labels.append(cls)
                    scores.append(conf)
                boxes = torch.Tensor(boxes)
                labels = torch.Tensor(labels)
                scores = torch.Tensor(scores)

                cur_result = {}
                cur_result["boxes"] = boxes
                cur_result["labels"] = labels
                cur_result["scores"] = scores

                result["{}".format(img_name)] = cur_result




        # Print time (inference-only)
        LOGGER.info(f"{s}{'' if len(det) else '(no detections), '}{dt[1].dt * 1E3:.1f}ms")

    ############################################## Print results 结果输出
    t = tuple(x.t / seen * 1E3 for x in dt)  # speeds per image
    LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {(1, 3, *imgsz)}' % t)




if __name__ == "__main__":

    run()















































