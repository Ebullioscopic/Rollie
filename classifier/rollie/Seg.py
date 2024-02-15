from ultralytics import YOLO

# Load a pre-trained YOLOv8n model
model = YOLO('yolov8n.pt')
names = model.model.names

# Perform inference on 'bus.jpg' with specified parameters with conf=0.5
results_generator = model.predict(source="0", verbose=False, conf=0.5, stream=True)

for results in results_generator:
    # Process detections
    boxes = results.boxes.xywh.cpu()
    clss = results.boxes.cls.cpu().tolist()
    confs = results.boxes.conf.float().cpu().tolist()

    for box, cls, conf in zip(boxes, clss, confs):
        print(f"Class Name: {names[int(cls)]}, Confidence Score: {conf}, Bounding Box: {box}")
        #output coordinates are given in the form of (x,y) of left top and width, height of bounding box
        #replace xywh with xyxy to get left top and right bottom coords
        #rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
        #rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
            #minSize=(30, 30), flags = cv2.CASCADE_SCALE_IMAGE)
