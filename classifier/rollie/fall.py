from ultralytics import YOLO

# Load a pre-trained YOLOv8n model
model = YOLO('fall_det_1.pt')
names = model.model.names

# Perform inference on 'bus.jpg' with specified parameters with conf=0.5
results_generator = model.predict(source="0", verbose=False, conf=0.5, stream=True, show=True)

for results in results_generator:
    # Process detections
    boxes = results.boxes.xywh.cpu()
    clss = results.boxes.cls.cpu().tolist()
    confs = results.boxes.conf.float().cpu().tolist()

    for box, cls, conf in zip(boxes, clss, confs):
        print(f"Class Name: {names[int(cls)]}, Confidence Score: {conf}, Bounding Box: {box}")