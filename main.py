import math
import numpy as np
from ultralytics import YOLO
import cv2
import cvzone
from sort import *

cap = cv2.VideoCapture('assets/vecteezy_traffic-BStock.mp4')
wd = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ht = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
mask = cv2.imread('assets/mask.png')
mask = cv2.resize(mask, (wd,ht))

model = YOLO('../YOLO-weights/yolov8s.pt')

#section
classNames = [
    "person", "bicycle", "car", "motorbike", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tvmonitor",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]
#endsection
set_count = set()

# Tracker
tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
limit = [100,297,1200,297]

#section for image
# res = model('image.jpg', show=True)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#endsection

while True:
    ret, image = cap.read()
    imgRegion = cv2.bitwise_and(image, mask)

    results = model(imgRegion, stream=True)

    detections = np.empty((0,5))

    for r in results:
        boxes = r.boxes
        for box in boxes:
            #bounding box
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2),int(y2)

            #confidence
            conf = math.ceil((box.conf[0]*100))/100
            origin = (max(0,x1),max(28,y1-10)) # handle overflow

            #class name
            cls= int(box.cls[0])
            if classNames[cls] == "car" and conf > 0.3:
                # cv2.rectangle(image,origin,(x1,y1),(0,255,0),2)
                # cv2.putText(image, f'{classNames[cls]} {conf}', origin, cv2.FONT_HERSHEY_SIMPLEX, 1, (255,0,255), 2)
                current_array = np.array([x1,y1,x2,y2,conf])
                detections = np.vstack((detections, current_array))

            # else:
            #     cv2.rectangle(image,(x1,y1),(x2,y2),(0,0,255),1)

    tracker_results = tracker.update(detections)
    cv2.line(image,(limit[0], limit[1]),(limit[2], limit[3]),(0,255,255),4)

    for result in tracker_results:
        x1,y1,x2,y2,Id = result
        x1,y1,x2,y2,Id = int(x1),int(y1),int(x2),int(y2),int(Id)
        w,h = x2-x1,y2-y1
        cv2.rectangle(image,(x1,y1),(x2,y2),(0,255,0),1)
        cv2.putText(image, f'{Id}', (max(0,x1), max(28,y1)), thickness=2, fontScale=1.5, fontFace=cv2.FONT_HERSHEY_SIMPLEX, color=(255,255,255))

        # tracker points
        cx,cy = x1+(w//2),y1+(h//2)
        cv2.circle(image,(cx,cy),5,(0,0,255),-1)

        if limit[0]<cx<limit[2] and limit[1]-10<cy<limit[3]+10:
            set_count.add(Id)
            cv2.line(image, (limit[0], limit[1]), (limit[2], limit[3]), (0, 255, 0), 4)
        cvzone.putTextRect(image, f'Cars : {len(set_count)}', (40,50), colorR=(49,40,34), thickness=2, colorT=(181,173,0))
    cv2.imshow('Detect',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()