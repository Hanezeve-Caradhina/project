import sys
from turtle import left
from typing import Dict
import cv2
import numpy as np
import stereoconfig
import math
from PIL import Image
import time
from queue import Queue
import argparse
from pathlib import Path
from detect import detect
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized
import threading
import BLE_SDK

# 程序运行状态 0:初始化, 1:拍照, 2:识别, 3:测距,
state = 0

counter = 0

left_photo = ''
right_photo = ''


class take_photos(threading.Thread):
    def __init__(self, queue, other_threads={}):
        threading.Thread.__init__(self)
        self.photo_queue = queue
        self.other_threads = other_threads
        self.camera = None
        # 用于暂停线程的标识
        self.__flag = threading.Event()
        self.__flag.set()  # 设置为True
        # 用于停止线程的标识
        self.__running = threading.Event()
        self.__running.set()  # 将running设置为True

    def pause(self):
        self.__flag.clear()  # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()  # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()  # 将线程从暂停状态恢复, 如果已经暂停的话
        self.__running.clear()  # 设置为False

    def run(self):
        utc = time.time()
        print("log---take_photo.run start!")

        self.camera = cv2.VideoCapture(0)

        # print("log---camera get over!!!")
        # 设置分辨率左右摄像机同一频率，同一设备ID；左右摄像机总分辨率2560x720；分割为两个1280x720
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            # now = time.time()
            # if now - utc >= interval:
            time.sleep(0.08)
            path1, path2 = self.take_photo()

            global state
            if state == 1 or photo_queue.qsize() < 3:
                self.resume()
            else:
                self.pause()

            self.__flag.wait()

            if photo_queue.qsize() >= 3:
                continue

            # 统计总循环时长
            # global circle_time
            # print("log-----------------------------------circle_time:", time.time() - circle_time, '\n\n')
            # circle_time = time.time()

            print("log---push photo start! state =", state)

            # utc = now
            print(path1)
            # 当队列中的图像小于两组，再压如新的左右图像
            if photo_queue.qsize() < 3:
                # print("press photo!")
                photo_queue.put(path1)
                photo_queue.put(path2)
                # print(photo_queue)
            # print(photo_queue)
            # print("get photo from queue!")
            # print(photo_queue)
            # print("log---------------push_photo_time:", time.time() - circle_time)
            print("log---push photo over! state =", 2)
            state = 2  # 进入识别模式
            self.other_threads['yolo'].resume()

    def take_photo(self):
        '''
        双目摄像头
        输出:拍摄的左右的jpg图片
        '''

        # print("log---come in take_photo!!!")

        global counter

        path1 = "imgs/left_" + str(counter) + ".jpg"
        path2 = "imgs/right_" + str(counter) + ".jpg"

        ret, frame = self.camera.read()

        # 裁剪坐标为[y0:y1, x0:x1]    HEIGHT * WIDTH
        left_frame = frame[0:480, 0:640]
        right_frame = frame[0:480, 0:640]

        # print("log---photo cut over!!!")
        cv2.imwrite(path1, left_frame)
        cv2.imwrite(path2, right_frame)
        counter += 1
        return path1, path2

    # with torch.no_grad():
    # if opt.update:  # update all models (to fix SourceChangeWarning)
    # for opt.weights in ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']:
    # detect()
    # strip_optimizer(opt.weights)
    # else:
    # obj_results=detect(path1,opt)
    # print(obj_results)
    # return obj_results


class YOLO(threading.Thread):
    def __init__(self, img_source_queue, det_queue, opt, save_img=False, other_threads={}):
        threading.Thread.__init__(self)
        self.det_queue = det_queue
        self.img_source_queue = img_source_queue
        self.my_opt = opt
        self.save_img = save_img
        self.first_photo_flag = 1
        self.other_threads = other_threads
        # 用于暂停线程的标识
        self.__flag = threading.Event()
        self.__flag.set()  # 设置为True
        # 用于停止线程的标识
        self.__running = threading.Event()
        self.__running.set()  # 将running设置为True

    def pause(self):
        self.__flag.clear()  # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()  # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()  # 将线程从暂停状态恢复, 如果已经暂停的话
        self.__running.clear()  # 设置为False

    def result(self, id, cls, pos):
        result = []
        cl = int(cls)
        p = [int(pos[0]), int(pos[1]), int(pos[2]), int(pos[3])]
        x_center = (p[2] - p[0]) / 2 + p[0]
        y_center = (p[3] - p[1]) / 2 + p[1]
        result = [id, cl, int(x_center), int(y_center)]
        return result

    def run(self):
        print("log---yolo.run start!")

        opt = self.my_opt

        img_source = self.img_source_queue.get()

        global left_photo, right_photo
        left_photo = img_source
        right_photo = self.img_source_queue.get()
        get_dis_photo_queue.put(left_photo)
        get_dis_photo_queue.put(right_photo)

        source, weights, view_img, save_txt, imgsz = img_source, opt.weights, opt.view_img, opt.save_txt, opt.img_size
        save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))

        # Directories
        save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
        (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA

        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
        if half:
            model.half()  # to FP16

        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once

        global state
        print("log---yolo init over!")
        print("log---yolo circle start!")
        while True:
            if state == 2 or ((not photo_queue.empty()) and (get_dis_photo_queue.qsize() < 3)):
                self.resume()
            else:
                self.pause()

            self.__flag.wait()

            if self.first_photo_flag == 0 and (photo_queue.empty() or (get_dis_photo_queue.qsize() >= 3)):
                continue

            print("log---yolo start! state = ", state)
            if self.first_photo_flag == 0:
                source = self.img_source_queue.get()
                left_photo = source
                right_photo = self.img_source_queue.get()
                get_dis_photo_queue.put(left_photo)
                get_dis_photo_queue.put(right_photo)
            else:
                self.first_photo_flag = 0

            # Set Dataloader
            vid_path, vid_writer = None, None
            if webcam:
                view_img = check_imshow()
                cudnn.benchmark = True  # set True to speed up constant image size inference
                dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            else:
                cudnn.benchmark = True  # new row to debug
                dataset = LoadImages(source, img_size=imgsz, stride=stride)

            # 下面注释部分与单张照片无关，放到循环外运行
            # # Get names and colors
            # names = model.module.names if hasattr(model, 'module') else model.names
            # colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
            #
            # # Run inference
            # if device.type != 'cpu':
            #     model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
            t0 = time.time()
            for path, img, im0s, vid_cap in dataset:

                # print("\nlog---yolo.run.path:", path)
                results = []
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)

                # Inference
                t1 = time_synchronized()
                pred = model(img, augment=opt.augment)[0]

                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes,
                                           agnostic=opt.agnostic_nms)
                t2 = time_synchronized()

                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)

                # Process detections
                for i, det in enumerate(pred):  # detections per image
                    if webcam:  # batch_size >= 1
                        p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                    else:
                        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

                    p = Path(p)  # to Path
                    save_path = str(save_dir / p.name)  # img.jpg
                    txt_path = str(save_dir / 'labels' / p.stem) + (
                        '' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                    s += '%gx%g ' % img.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh

                    if len(det):

                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                        # Print results
                        for c in det[:, -1].unique():
                            n = (det[:, -1] == c).sum()  # detections per class
                            s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        # 输出检测对象属性列表
                        obj_id = 0
                        # Write results
                        for *xyxy, conf, cls in reversed(det):
                            # print(xyxy,conf,cls)
                            if save_txt:  # Write to file
                                xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(
                                    -1).tolist()  # normalized xywh
                                line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                with open(txt_path + '.txt', 'a') as f:
                                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

                            if save_img or view_img:  # Add bbox to image
                                label = f'{names[int(cls)]} {conf:.2f}'
                                plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=3)

                            # print("log---yolo.run.obj_id", obj_id)
                            # print("log---yolo.run.cls", cls)
                            # print("log---yolo.run.xyxy", xyxy)
                            results.append(self.result(id=obj_id, cls=cls, pos=xyxy))
                            obj_id += 1

                    # Print time (inference + NMS)
                    print(f'{s}Done. ({t2 - t1:.3f}s)')
                    self.det_queue.put(results)

                    break
                break
                #         # Stream results
            #         if view_img:
            #             cv2.imshow(str(p), im0)
            #             cv2.waitKey(1)  # 1 millisecond
            #
            #         # Save results (image with detections)
            #         if save_img:
            #             if dataset.mode == 'image':
            #                 cv2.imwrite(save_path, im0)
            #             else:  # 'video' or 'stream'
            #                 if vid_path != save_path:  # new video
            #                     vid_path = save_path
            #                     if isinstance(vid_writer, cv2.VideoWriter):
            #                         vid_writer.release()  # release previous video writer
            #                     if vid_cap:  # video
            #                         fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #                         w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #                         h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #                     else:  # stream
            #                         fps, w, h = 30, im0.shape[1], im0.shape[0]
            #                         save_path += '.mp4'
            #                     vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #                 vid_writer.write(im0)
            # if save_txt or save_img:
            #     s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
            #     print(f"Results saved to {save_dir}{s}")

            print(f'Done. ({time.time() - t0:.3f}s)')
            # global circle_time
            # print("log---------------yolo_time:", time.time() - circle_time)
            print("log---yolo over! state = ", 3)
            state = 3
            self.other_threads['distance'].resume()
            self.other_threads['take_photo'].resume()


class get_distance(threading.Thread):
    def __init__(self, det_queue, other_threads={}):
        threading.Thread.__init__(self)
        self.det_queue = det_queue
        self.other_threads = other_threads
        # 用于暂停线程的标识
        self.__flag = threading.Event()
        self.__flag.set()  # 设置为True
        # 用于停止线程的标识
        self.__running = threading.Event()
        self.__running.set()  # 将running设置为True

    def pause(self):
        self.__flag.clear()  # 设置为False, 让线程阻塞

    def resume(self):
        self.__flag.set()  # 设置为True, 让线程停止阻塞

    def stop(self):
        self.__flag.set()  # 将线程从暂停状态恢复, 如果已经暂停的话
        self.__running.clear()  # 设置为False

    def run(self):
        print("log---get_distance.run start!")
        global state
        while True:
            if not get_dis_photo_queue.empty() and not self.det_queue.empty():
                self.resume()
            else:
                self.pause()

            self.__flag.wait()

            print("log---get_dis start! state = ", state)

            if True:
                obj_list = self.det_queue.get()
                get_dis_left_photo = get_dis_photo_queue.get()
                get_dis_right_photo = get_dis_photo_queue.get()
                distance = self.get_dis_SGBM(obj_list, get_dis_left_photo, get_dis_right_photo)

                print("get_dis_path", get_dis_left_photo)
                print("距离为:", distance)

                result = self.calculator(obj_list,distance)

            global FPS_time, FPS_time_10, FPS_10_cnt
            print("log--------------------------------------------------FPS = ", 1 / (time.time() - FPS_time))
            FPS_time = time.time()

            FPS_10_cnt += 1
            if FPS_10_cnt == 10:
                print("log$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$FPS_10 = ",
                      FPS_10_cnt / (time.time() - FPS_time_10))
                FPS_time_10 = time.time()
                FPS_10_cnt = 0

            global circle_time
            # print("log---------------get_dis_time:", ti me.time() - circle_time)
            print("log---get_dis over! state = ", 1)
            state = 1
            self.other_threads['take_photo'].resume()
            self.other_threads['yolo'].resume()

            # 通过屏幕向蓝牙发信息
            commCtrl.packetSend(result)

    def get_dis_SGBM(self, obj_list, lphoto, rphoto):
        '''
        测距模块
        输入:目标物体的,左右图
        输出:距离,单位米
        '''
        # 读取图片
        iml = cv2.imread(lphoto)  # 左图
        imr = cv2.imread(rphoto)  # 右图

        # 将BGR格式转换成灰度图片
        imgL = cv2.cvtColor(iml, cv2.COLOR_BGR2GRAY)
        imgR = cv2.cvtColor(imr, cv2.COLOR_BGR2GRAY)

        # 读取相机内参和外参
        config = stereoconfig.stereoCamera()
        config.__init__()

        # 校正查找映射表,将原始图像和校正后的图像上的点一一对应起来
        height, width = iml.shape[0:2]
        R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(config.cam_matrix_left,
                                                                          config.distortion_l,
                                                                          config.cam_matrix_right,
                                                                          config.distortion_r,
                                                                          (width, height),
                                                                          config.R,
                                                                          config.T)
        left_map1, left_map2 = cv2.initUndistortRectifyMap(config.cam_matrix_left, config.distortion_l, R1, P1,
                                                           (width, height), cv2.CV_16SC2)
        right_map1, right_map2 = cv2.initUndistortRectifyMap(config.cam_matrix_right, config.distortion_r, R2, P2,
                                                             (width, height), cv2.CV_16SC2)

        # cv2.remap 重映射，就是把一幅图像中某位置的像素放置到另一个图片指定位置的过程。
        # 依据MATLAB测量数据重建无畸变图片
        img1_rectified = cv2.remap(imgL, left_map1, left_map2, cv2.INTER_LINEAR)
        img2_rectified = cv2.remap(imgR, right_map1, right_map2, cv2.INTER_LINEAR)

        blockSize = 8
        img_channels = 1
        stereo = cv2.StereoSGBM_create(minDisparity=1,
                                       numDisparities=16,
                                       blockSize=blockSize,
                                       P1=8 * img_channels * blockSize * blockSize,
                                       P2=32 * img_channels * blockSize * blockSize,
                                       disp12MaxDiff=-1,
                                       preFilterCap=1,
                                       uniquenessRatio=10,
                                       speckleWindowSize=100,
                                       speckleRange=100,
                                       mode=cv2.STEREO_SGBM_MODE_HH4)
        # 计算视差
        disparity = stereo.compute(img1_rectified, img2_rectified)
        # 计算三维坐标数据值
        threeD = cv2.reprojectImageTo3D(disparity, Q, handleMissingValues=True)
        threeD = threeD * 16

        distance = []
        temp_dis = 0
        cnt = 0
        for i in range(len(obj_list)):
            middle_x = obj_list[i][2]
            middle_y = obj_list[i][3]

            for m in range(-2, 3):
                for n in range(-2, 3):
                    if ((middle_x - m) < 0 or (middle_y - n) < 0 or (middle_x - m) >= width or (
                            middle_y - n) >= height):
                        continue
                    if (threeD[middle_y - n][middle_x - m][0] != -float('inf') and threeD[middle_y - n][middle_x - m][
                        0] != float('inf')):
                        cnt += 1
                        temp_dis = temp_dis + math.sqrt(
                            threeD[middle_y - n][middle_x - m][0] ** 2
                            + threeD[middle_y - n][middle_x - m][1] ** 2
                            + threeD[middle_y - n][middle_x - m][2] ** 2)
            if (temp_dis != 0):
                temp_dis = temp_dis / cnt / 1000.0  # mm -> m
                distance.append(temp_dis)

        return distance


    def calculator(self,obj_results, distance):
        z1 = [1.79200062e-07, -2.85637151e-04, 2.02063937e-01, -5.68983294e-02]
        result = {}
        temp_dir = {}
        for i in range(len(distance)):
            temp_dir.clear()
            x = obj_results[i][2]
            camera_x = 470
            x1 = abs(x - camera_x)
            angle = z1[0] * x1 * x1 * x1 + z1[1] * x1 * x1 + z1[2] * x1 + z1[3]
            xval = distance[i] * np.sin(angle / 180)
            yval = distance[i] * np.cos(angle / 180)
            # temp_dir = {'NAME': obj_results[i][0], 'CAT': obj_results[i][1],
            #             'XVAL': xval, 'YVAL': yval,
            #             'DIS': distance[i], 'ANG': angle}
            temp_dir = BLE_SDK.vehicle(obj_results[i][0],obj_results[i][1],distance[i],angle,0,0)
            result[i] = temp_dir
            # print(result)
        return result

if __name__ == '__main__':
    print("log---init start! state =", state)

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='weights/v5lite-e.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='sample', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=480, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', default=False, action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', default=True, action='store_true', help='do not save images/videos')
    # 0 person, 1 bicycle, 2 car, 3 motorcycle, 5 bus, 7 truck
    parser.add_argument('--classes', default=[0, 1, 2, 3, 5, 7], nargs='+', type=int,
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    print("yolo_opt\n", opt)
    check_requirements(exclude=('pycocotools', 'thop'))

    commCtrl = BLE_SDK.commControlPanel()

    print("log---init over! state = ", 1)
    state = 1

    # interval:拍照间隔，单位秒 e.g.:2为2s
    interval = 0
    # 照片队列,每次同时存入左右图,也同时读出左右图
    photo_queue = Queue()

    # 专为测距准备的相片队列
    get_dis_photo_queue = Queue()

    det_queue = Queue()

    take_photo = take_photos(photo_queue)
    yolo = YOLO(photo_queue, det_queue, opt)
    distance = get_distance(det_queue, {'take_photo': take_photo, 'yolo': yolo})

    take_photo.other_threads = {'distance': distance, 'yolo': yolo}
    yolo.other_threads = {'distance': distance, 'take_photo': take_photo}

    myThreads = []
    myThreads.append(take_photo)
    myThreads.append(yolo)
    myThreads.append(distance)

    # 用于统计总循环时长
    circle_time = time.time()

    # 用于统计帧率
    FPS_time = time.time()
    FPS = 0

    FPS_time_10 = 0
    FPS_10 = 0
    FPS_10_cnt = 0

    for i in range(len(myThreads)):
        myThreads[i].start()

