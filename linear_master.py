import argparse
import os
import shutil

import sys
import time
import datetime
from multiprocessing.pool import ThreadPool
from queue import Queue
from variables import args
import cv2  # Optional, see below
import numpy as np
import pandas as pd
import torch
import linecache
import torch.backends.cudnn as cudnn

from layers.output_utils import postprocess
from utils.augmentations import FastBaseTransform

from data import set_cfg
from yolact import Yolact


from data import config
from data.config import cfg as linear_config

# from processing import dataProcessing


noError = None
position_df = None
errorVideoFlag = False

def PrintException():  # For printing the exception in required format
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    error_text = str(datetime.datetime.now()) + f' EXCEPTION IN ({filename}, LINE {lineno} "{line.strip()}"): {exc_obj}'
    print(error_text)
    # logger_obj.logger.error(error_text)
class Detections:

    def __init__(self):
        self.frame_queue = Queue()
        self.frame_queue_display = Queue()
        self.lat_long_queue = Queue()
        self.frames_skipped = args['frames_skipped']
        self.frame_count = 0
        self.FrameCount_list = Queue()
        self.cap = cv2.VideoCapture(video_path)
        self.cap2 = cv2.VideoCapture(video_path)
        self.vsize =None
        self.quit = False
        # self.setup()
        self.classes_queue = Queue()
        self.scores_queue = Queue()
        self.boxes_queue = Queue()
        self.masks_queue = Queue()
        # self.frame_queue_masks = Queue()
        self.detection_df = pd.DataFrame(columns=['Frame', 'Assets_LHS', 'Assets_RHS'])
        self.detection_dict = {}
        self.position_df = pd.DataFrame(columns=['Frame', 'Position', 'Speed'])
        self.video_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.totalFrames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        self.class_names = ""
        output_video = "Output_" + str(video_name) + "_linear.MP4"
        output_video_size = (1280, 720)
        self.asset_list=args['Assets']
        self.t = time.time()


    def process_timer(self, signum, frame):
        global errorVideoFlag
        self.quit = True
        errorVideoFlag = True

    def set_saved_video(self, output_video):
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        video = cv2.VideoWriter(self.arguments['video_output_path'] + output_video, fourcc, fps, (800, 448))
        return video

    def video_capture(self):
        global errorVideoFlag
        try:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret or self.quit:
                    break
                # print(self.frame_queue.qsize())
                while self.frame_queue.qsize() > 100:
                    if not self.quit:
                        time.sleep(0.001)
                    else:
                        break
                # print(self.frame_queue.qsize())
                # print(self.frame_count)
                if self.vsize is None:
                    self.vsize = frame.shape
                if self.frame_count % self.frames_skipped == 0:
                    frame = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)
                    self.frame_queue.put(frame)
                    self.FrameCount_list.put(self.frame_count)
                self.frame_count += 1
                # print(self.frame_count)
            self.cap.release()
            print('done capturing')
        except Exception as e:
            self.cap.release()
            print("##############################################")
            print("Exception in Video Capture", e)

            errorVideoFlag = True


            print("##############################################")

    def inference(self):
        global errorVideoFlag
        torch.backends.cudnn.fastest = True
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        # Use whichever config you want here.
        try:
            set_cfg(args["config"])
            print("=======")
            print('Loading model...', args["weight"])
            net = Yolact()
            net.load_weights(args["weight"])
            net.eval()
            print('Model Loading Done.')
            net.detect.use_fast_nms = True
            net.detect.use_cross_class_nms = False
            self.class_names = linear_config.dataset.class_names

        except Exception as ex:

            print("Model not loaded properly")
            print("Exception while loading model, ", str(ex))
            self.quit = True

        try:
            while True:


                start_time = time.time()
                if self.quit:
                    break
                try:
                    img = self.frame_queue.get(timeout=5)
                except Exception:
                    print('inferring timeout')

                    self.quit = True
                    break
                # print(img.shape)
                frame = torch.from_numpy(img).cuda().float()
                batch = FastBaseTransform()(frame.unsqueeze(0))
                preds = net(batch)

                t = postprocess(preds, frame.shape[1], frame.shape[0],
                                score_threshold=args["linear_detection_threshold"])

                idx = t[1].argsort(0, descending=True)[:10]


                classes, scores, boxes = [x[idx].detach().cpu().numpy() for x in t[:3]]


                self.classes_queue.put(classes)
                self.boxes_queue.put(boxes)
                self.scores_queue.put(scores)
                self.frame_queue_display.put(img)

                fps = int(1 / (time.time() - start_time))

        except Exception as ex:
            print("##################################")
            print(ex)
            self.quit = True
            errorVideoFlag = True
            print("##################################")
    def drawing(self):
        json={y+x:[0,None] for y in ['LEFT_','RIGHT_'] for x in args['Assets'].keys()}
        ann_json={x:0  for x in args['Assets'].keys()}
        global errorVideoFlag
        while True:
            time.sleep(2)
            if self.frame_queue_display.qsize() > 1:
                break
        try:
            df_idx = 0
            frame_count = 0
            cv2.namedWindow(video_name, cv2.WINDOW_NORMAL)

            while True:
                start_time = time.time()
                if self.quit:
                    break
                try:
                    classes = self.classes_queue.get(timeout=5)
                    boxes = self.boxes_queue.get(timeout=5)
                    scores = self.scores_queue.get(timeout=5)
                    img = self.frame_queue_display.get(timeout=5)
                    # print(classes.shape,boxes.shape,scores.shape)
                except Exception:
                    print('drawing points timeout')
                    break

                json[frame_count] = {}
                ann_json[frame_count] = {}
                for i in range(len(boxes)):
                    # print(scores[i])

                    asset_name = self.class_names[classes[i]]
                    if scores[i] < args['Assets'][asset_name]:
                        continue
                    # print(boxes[i])
                    # x1, y1, x2, y2 = boxes[i]
                    # print(x1, y1, x2, y2)
                    (x1, y1, x2, y2) = boxes[i]
                    qx1=int(x1*self.vsize[1]/1280)
                    qx2=int(x2*self.vsize[1]/1280)
                    qy1=int(y1*self.vsize[0]/720)
                    qy2=int(y2*self.vsize[0]/720)

                    if (x1 + x2) / 2 > img.shape[1] / 2:  # Right assets
                        if 'RIGHT_'+asset_name not in json[frame_count]:
                            if json['RIGHT_' + asset_name][1] is not None and frame_count - json['RIGHT_' + asset_name][
                                1] > 50:
                                json['RIGHT_' + asset_name][0] += 1
                                ann_json[asset_name] += 1
                            json['RIGHT_' + asset_name][1] = frame_count
                            json[frame_count]['RIGHT_'+asset_name]=[json['RIGHT_'+asset_name][0],[qx1, qy1], [qx2, qy2]]
                            if asset_name not in ann_json[frame_count]:
                                ann_json[frame_count][asset_name]=[]
                            ann_json[frame_count][asset_name].append([str(ann_json[asset_name]),[qx1, qy1], [qx2, qy2]])

                            cv2.putText(img, str(asset_name)+str(json['RIGHT_'+asset_name][0]), (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_TRIPLEX, 0.8,
                                        (0, 255, 0), 1, cv2.LINE_AA)


                    else:
                        if 'LEFT_'+asset_name not in json[frame_count]:

                            if json['LEFT_' + asset_name][1] is not None and frame_count - json['LEFT_' + asset_name][1] >50:
                                json['LEFT_'+asset_name][0]+=1
                                ann_json[asset_name] += 1
                            json['LEFT_' + asset_name][1] = frame_count
                            json[frame_count]['LEFT_'+asset_name]=[json['LEFT_'+asset_name][0],[qx1, qy1], [qx2, qy2]]
                            if asset_name not in ann_json[frame_count]:
                                ann_json[frame_count][asset_name]=[]
                            ann_json[frame_count][asset_name].append([str(ann_json[asset_name]),[qx1, qy1], [qx2, qy2]])
                            cv2.putText(img, str(asset_name)+str(json['LEFT_'+asset_name][0]), (int(x1), int(y1) - 25), cv2.FONT_HERSHEY_TRIPLEX, 0.8,
                                        (0, 255, 0), 1, cv2.LINE_AA)


                    cv2.putText(img, str(round(scores[i], 2)), (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_TRIPLEX,
                                0.8, (0, 255, 0), 1, cv2.LINE_AA)
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)


                cv2.putText(img, str(frame_count) , (700, 170), cv2.FONT_HERSHEY_TRIPLEX,
                            0.8, (0, 255, 255), 1, cv2.LINE_AA)
                percentage_completed = (frame_count / self.totalFrames)
                img[-7:-3, :(int(percentage_completed * img.shape[1])), 2] = 255


                frame_count += self.frames_skipped
                df_idx += 1
                fps = int(1 / (time.time() - start_time))
                fps=f"fps : {fps}"
                cv2.putText(img, fps, (500, 70), cv2.FONT_HERSHEY_TRIPLEX, 0.3,
                            (0, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(video_name, img)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.quit = True
                    break
            cv2.destroyAllWindows()


        except Exception as ex:
            print("##################################")
            print(ex)
            PrintException()
            errorVideoFlag = True
            self.quit = True
            cv2.destroyAllWindows()

            print("##################################")
        import json
        for x in args['Assets'].keys():
            ann_json[x]+=1 
        with open(args['video'].replace(".MP4",".json"),"w") as f:
            f.write(json.dumps(ann_json))
        return self.detection_df


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--video", required=True,
                    help="path to input video")

    args["video"] = vars(ap.parse_args())['video']
    # print(vars(ap.parse_args()))
    print(args)
    global video_id, fromMetadata

    video_path = args["video"]
    print(video_path)
    video_name = os.path.basename(video_path).split(".")[0]

    ####################### Timer for total processing ########################
    cap = cv2.VideoCapture(video_path)
    video_fps = int(cap.get(cv2.CAP_PROP_FPS))
    totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    error_message = None
    Dfs = []

    try:
        video_time = int(totalFrames / video_fps)
    except Exception as e:

        cap.release()
        print("video_error")

        sys.exit()
        
    ############################# Code Ends #################################

    # print("gps noError from new safecam :", noError)


    yolact = Detections()
    pool = ThreadPool(processes=3)
    t1 = pool.apply_async(yolact.video_capture)
    t2 = pool.apply_async(yolact.inference)
    t3 = pool.apply_async(yolact.drawing)
    t1.get()
    t2.get()
    detection_df = t3.get()


    # sys.exit()

