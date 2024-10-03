from ultralytics import YOLO
model = YOLO("yolo11s.pt")

results = model.train(data = 'data.yaml', epochs = 10, imgsz=640)

