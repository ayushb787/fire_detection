from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
# import tensorflow as tf
import re

# print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

video_path = 0
# video_path = 'sample_1.mp4'
cap = cv2.VideoCapture(video_path)
model = YOLO('fire.pt')
while cap.isOpened():
    success, frame = cap.read()

    if success:
        result = model.predict(source=frame, stream_buffer=False, conf=0.6, vid_stride=0, show_labels=True, show_conf=True)
        for r in result:
            arr = r.boxes.xyxy
            im_bgr = r.plot()  # BGR-order numpy array
            im_rgb = Image.fromarray(im_bgr[..., ::-1])
            # im_rgb.show()
        annotated_frame = result[0].plot()
        annotated_frame = cv2.resize(annotated_frame, (640, 640))
        cv2.imshow("YOLOv8 Inference", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()