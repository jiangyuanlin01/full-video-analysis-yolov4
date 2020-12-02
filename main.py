from __future__ import division, print_function, absolute_import
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image

from yolo import YOLO
from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from keras import backend

backend.clear_session()
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "test_video/video.avi")
ap.add_argument("-c", "--class",help="name of class",default = "person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")

def main(yolo):
    start = time.time()
    #Definition of the parameters
    max_cosine_distance = 0.9
    nn_budget = None
    nms_max_overlap = 0.3 #非极大抑制的阈值

    counter1 = []
    counter2 = []
    counter3 = []
    counter4 = []
    counter5 = []
    counter6 = []
    counter7 = []
    counter8 = []
    counter9 = []
    counter10 = []
    counter1x = 0
    counter2x = 0
    counter3x = 0
    counter4x = 0
    counter5x = 0
    counter6x = 0
    counter7x = 0
    counter8x = 0
    counter9x = 0
    counter10x = 0

    #deep_sort
    model_filename = 'model_data/market1501.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)
    writeVideo_flag = True
    #video_path = "../../yolo_dataset/t1_video/test_video/det_t1_video_00025_test.avi"
    video_capture = cv2.VideoCapture(args["input"])
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('output/output.avi', fourcc, 30, (w, h))
        list_file = open('detection.txt', 'w')
        frame_index = -1

    fps = 0.0

    while True:
        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
        t1 = time.time()

       # image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        boxs,class_names = yolo.detect_image(image)
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]
        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        i1 = int(0)
        i2 = int(0)
        i3 = int(0)
        i4 = int(0)
        i5 = int(0)
        i6 = int(0)
        i7 = int(0)
        i8 = int(0)
        i9 = int(0)
        i10 = int(0)
        indexIDs = []
        c = []
        boxes = []
        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)

        for track, class_name in zip(tracker.tracks, class_names):
            if not track.is_confirmed() or track.time_since_update > 1:
                continue

            # boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            print("relal class:" + class_name[0])
            # 分别保存每个类别的track_id
            if class_name == ['person']:
                counter1.append(int(track.track_id))
                count1 = len(set(counter1))
                print(int(track.track_id))
                print(str(count1))
                print(counter1x)
                if not os.path.isdir("output/person"):
                   os.makedirs("output/person")
                if counter1x != count1:
                   save1image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                   if save1image.size != 0:
                        cv2.imwrite("output/person" + "/" + str(count1) + ".jpeg", save1image, [int(cv2.IMWRITE_JPEG_QUALITY), 100])
                   cv2.waitKey(20)
                   print("执行")
                   print(int(bbox[1]))
                   print(int(bbox[3]))
                   print(int(bbox[0]))
                   print(int(bbox[2]))
                   counter1x = count1
            if class_name == ['bicycle']:
                counter2.append(int(track.track_id))
                count2 = len(set(counter2))
                print(int(track.track_id))
                print(str(count2))
                print(counter2x)
                if not os.path.isdir("output/bicycle"):
                    os.makedirs("output/bicycle")
                if counter2x != count2:
                    save1image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    if save1image.size != 0:
                        cv2.imwrite("output/bicycle" + "/" + str(count2) + ".jpeg", save1image,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    cv2.waitKey(20)
                    print("执行")
                    print(int(bbox[1]))
                    print(int(bbox[3]))
                    print(int(bbox[0]))
                    print(int(bbox[2]))
                    counter2x = count2
            if class_name == ['car']:
                counter3.append(int(track.track_id))
                count3 = len(set(counter3))
                print(int(track.track_id))
                print(str(count3))
                print(counter3x)
                if not os.path.isdir("output/car"):
                    os.makedirs("output/car")
                if counter3x != count3:
                    save1image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    if save1image.size != 0:
                        cv2.imwrite("output/car" + "/" + str(count3) + ".jpeg", save1image,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    cv2.waitKey(20)
                    print("执行")
                    print(int(bbox[1]))
                    print(int(bbox[3]))
                    print(int(bbox[0]))
                    print(int(bbox[2]))
                    counter3x = count3
            if class_name == ['motorbike']:
                counter4.append(int(track.track_id))
                count4 = len(set(counter4))
                print(int(track.track_id))
                print(str(count4))
                print(counter4x)
                if not os.path.isdir("output/motorbike"):
                    os.makedirs("output/motorbike")
                if counter4x != count4:
                    save1image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    if save1image.size != 0:
                        cv2.imwrite("output/motorbike" + "/" + str(count4) + ".jpeg", save1image,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    cv2.waitKey(20)
                    print("执行")
                    print(int(bbox[1]))
                    print(int(bbox[3]))
                    print(int(bbox[0]))
                    print(int(bbox[2]))
                    counter4x = count4
            if class_name == ['bus']:
                counter5.append(int(track.track_id))
                count5 = len(set(counter5))
                print(int(track.track_id))
                print(str(count5))
                print(counter5x)
                if not os.path.isdir("output/bus"):
                    os.makedirs("output/bus")
                if counter5x != count5:
                    save1image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    if save1image.size != 0:
                        cv2.imwrite("output/bus" + "/" + str(count5) + ".jpeg", save1image,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    cv2.waitKey(20)
                    print("执行")
                    print(int(bbox[1]))
                    print(int(bbox[3]))
                    print(int(bbox[0]))
                    print(int(bbox[2]))
                    counter5x = count5
            if class_name == ['truck']:
                counter6.append(int(track.track_id))
                count6 = len(set(counter6))
                print(int(track.track_id))
                print(str(count6))
                print(counter6x)
                if not os.path.isdir("output/truck"):
                    os.makedirs("output/truck")
                if counter6x != count6:
                    save1image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    if save1image.size != 0:
                        cv2.imwrite("output/truck" + "/" + str(count6) + ".jpeg", save1image,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    cv2.waitKey(20)
                    print("执行")
                    print(int(bbox[1]))
                    print(int(bbox[3]))
                    print(int(bbox[0]))
                    print(int(bbox[2]))
                    counter6x = count6
            if class_name == ['stop sign']:
                counter7.append(int(track.track_id))
                count7 = len(set(counter7))
                print(int(track.track_id))
                print(str(count7))
                print(counter7x)
                if not os.path.isdir("output/stop sign"):
                    os.makedirs("output/stop sign")
                if counter7x != count7:
                    save1image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    if save1image.size != 0:
                        cv2.imwrite("output/stop sign" + "/" + str(count7) + ".jpeg", save1image,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    cv2.waitKey(20)
                    print("执行")
                    print(int(bbox[1]))
                    print(int(bbox[3]))
                    print(int(bbox[0]))
                    print(int(bbox[2]))
                    counter7x = count7
            if class_name == ['traffic light']:
                counter8.append(int(track.track_id))
                count8 = len(set(counter8))
                print(int(track.track_id))
                print(str(count8))
                print(counter8x)
                if not os.path.isdir("output/traffic light"):
                    os.makedirs("output/traffic light")
                if counter8x != count8:
                    save1image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
                    if save1image.size != 0:
                        cv2.imwrite("output/traffic light" + "/" + str(count8) + ".jpeg", save1image,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    cv2.waitKey(20)
                    print("执行")
                    print(int(bbox[1]))
                    print(int(bbox[3]))
                    print(int(bbox[0]))
                    print(int(bbox[2]))
                    counter8x = count8
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            bbox = track.to_tlbr()
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            # 显示类别
            cv2.putText(frame, str(class_name), (int(bbox[0]), int(bbox[1] - 20)), 0, 5e-3 * 150, (color), 2)
            # 当前画面中的每个类别单独计数
            if class_name == ['person']:
                i1 = i1 +1
            if class_name == ['bicycle']:
                i2 = i2 +1
            if class_name == ['car']:
                i3 = i3 + 1
            if class_name == ['motorbike']:
                i4 = i4 + 1
            if class_name == ['bus']:
                i5 = i5 + 1
            if class_name == ['truck']:
                i6 = i6 + 1
            if class_name == ['stop sign']:
                i7 = i7 + 1
            if class_name == ['traffic light']:
                i7 = i7 + 1

            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]
            pts[track.track_id].append(center)
            thickness = 5
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)

	        # draw motion path 移动路径
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        # 统计每类物品的总数
        count1 = len(set(counter1))
        count2 = len(set(counter2))
        count3 =len(set(counter3))
        count4 = len(set(counter4))
        count5 = len(set(counter5))
        count6 =len(set(counter6))
        count7 = len(set(counter7))
        count8 = len(set(counter8))
        cv2.putText(frame, "Total person Counter: "+str(count1),(int(20), int(120)),0, 5e-3 * 100, (0,255,0),2)
        cv2.putText(frame, "Current person Counter: "+str(i1),(int(20), int(100)),0, 5e-3 * 100, (0,255,0),2)
        cv2.putText(frame, "Total bicycle Counter: "+str(count2),(int(20), int(80)),0, 5e-3 * 100, (0,255,0),2)
        cv2.putText(frame, "Current bicycle Counter: "+str(i2),(int(20), int(60)),0, 5e-3 * 100, (0,255,0),2)
        cv2.putText(frame, "Total car Counter: " + str(count3), (int(20), int(140)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Current car Counter: " + str(i3), (int(20), int(160)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Total motorbike Counter: " + str(count4), (int(20), int(180)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Current motorcycle Counter: " + str(i4), (int(20), int(200)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Total bus Counter: " + str(count5), (int(20), int(180)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Current bus Counter: " + str(i5), (int(20), int(200)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Total truck Counter: " + str(count6), (int(20), int(180)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Current truck Counter: " + str(i6), (int(20), int(200)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Total stop sign Counter: " + str(count7), (int(20), int(180)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Current stop sign Counter: " + str(i7), (int(20), int(200)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Total traffic light Counter: " + str(count8), (int(20), int(180)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "Current traffic light Counter: " + str(i8), (int(20), int(200)), 0, 5e-3 * 100, (0, 255, 0), 2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 100, (0,255,0),3)
        # cv2.namedWindow("YOLO3_Deep_SORT", 0);
        # cv2.resizeWindow('YOLO3_Deep_SORT', 1024, 768);
        # cv2.imshow('YOLO3_Deep_SORT', frame)
        if writeVideo_flag:
            #save a frame
            out.write(frame)
            frame_index = frame_index + 1
            list_file.write(str(frame_index)+' ')

            if len(boxs) != 0:
                for i in range(0,len(boxs)):
                    list_file.write(str(boxs[i][0]) + ' '+str(boxs[i][1]) + ' '+str(boxs[i][2]) + ' '+str(boxs[i][3]) + ' ')
            list_file.write('\n')
        fps = ( fps + (1./(time.time()-t1)) ) / 2
        #print(set(counter))
        cv2.namedWindow("test", cv2.WINDOW_NORMAL)
        cv2.imshow("test", frame)
        cv2.waitKey(1)
        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

    if len(pts[track.track_id]) != None:
       print(args["input"][-13:-4] + ": " + str(count1) + " " + 'person Found')
       print(args["input"][-13:-4] + ": " + str(count2) + " " + 'bicycle Found')
       print(args["input"][-13:-4] + ": " + str(count3) + " " + 'car Found')
       print(args["input"][-13:-4] + ": " + str(count4) + " " + 'motorbike Found')
       print(args["input"][-13:-4] + ": " + str(count5) + " " + 'bus Found')
       print(args["input"][-13:-4] + ": " + str(count6) + " " + 'truck Found')
       print(args["input"][-13:-4] + ": " + str(count7) + " " + 'stop sign Found')
       print(args["input"][-13:-4] + ": " + str(count7) + " " + 'traffic light Found')
       resulttxt = open("检测结果汇总.txt", 'w+')
       resulttxt.write("people-")  # 写入
       resulttxt.write(str(count1))  # 写入
       resulttxt.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
       resulttxt.write("bicycle-")  # 写入
       resulttxt.write(str(count2))  # 写入
       resulttxt.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
       resulttxt.write("car-")  # 写入
       resulttxt.write(str(count3))  # 写入
       resulttxt.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
       resulttxt.write("motorbike-")  # 写入
       resulttxt.write(str(count4))  # 写入
       resulttxt.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
       resulttxt.write("bus-")  # 写入
       resulttxt.write(str(count5))  # 写入
       resulttxt.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
       resulttxt.write("truck-")  # 写入
       resulttxt.write(str(count6))  # 写入
       resulttxt.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
       resulttxt.write("stop sign-")  # 写入
       resulttxt.write(str(count7))  # 写入
       resulttxt.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
       resulttxt.write("traffic light-")  # 写入
       resulttxt.write(str(count7))  # 写入
       resulttxt.write('\n')  # 有时放在循环里面需要自动转行，不然会覆盖上一条数据
    else:
       print("[No Found]")

    video_capture.release()

    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main(YOLO())
