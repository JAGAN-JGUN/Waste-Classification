import cv2
import numpy as np
import time
import Predict as pr
# import Predict.py as pr

net = cv2.dnn.readNet("Obdata\yolov3.weights","Obdata\yolov3.cfg")
classes = []
with open("Obdata\coco.names.txt","r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
outputlayers= [layer_names[i-1] for i in net.getUnconnectedOutLayers()]
# a = np.random.randint(0,255)
# b = np.random.randint(0,255)
# c = np.random.randint(0,255)
# color = (a,b,c)
url = ""

Stream = cv2.VideoCapture(url)
# font = cv2.FONT_HERSHEY_PLAIN
# starting_time = time.time()
frame_id = 0

while True:
    _,frame = Stream.read()
    height, width, channels = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (640,640), (0,0,0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(outputlayers)

    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                
                boxes.append([x,y,w,h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.6)

    if len(indexes) > 0:
        cv2.waitKey(10)
        _,frame2 = Stream.read()
        frame2 = cv2.resize(frame2,dsize=(256,256))
        frame2 = cv2.cvtColor(frame2,cv2.COLOR_BGR2RGB)
        # predictions(frame2)
        pr.predictions(frame2)
        while True:
            data = pr.IOT.readline()[:2]
            if data:
                break

    # for i in range(len(boxes)):
    #     if i in indexes:
    #         x,y,w,h = boxes[i]
    #         label = str(classes[class_ids[i]])
    #         confidence = confidences[i]
    #         cv2.rectangle(frame, (int(x),int(y)), (int(x+w),int(y+h)), color, 2)
    #         cv2.putText(frame, label+" "+str(round(confidence,2)), (x+5,y+30), font, 1, color, 2)
    
    # elapsed_time = time.time() - starting_time
    # fps = frame_id/elapsed_time
    # cv2.putText(frame,"FPS:"+str(round(fps,2)),(10,50),font,2,(0,0,0),1)
    # cv2.imshow("Image",frame)
    key = cv2.waitKey(1)
    frame_id+=1
    if key == 27:
        break

Stream.release()
cv2.destroyAllWindows()