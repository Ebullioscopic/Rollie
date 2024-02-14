from ultralytics import YOLO

model = YOLO('yolov8n.pt')

results = model(source=0, show=True, conf=0.4, save=False,verbose=False,stream=True)
#set verbose to true to get printed output and change save to save the annotated output