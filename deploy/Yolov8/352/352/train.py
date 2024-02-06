from ultralytics import YOLO

# Load a model
model = YOLO('yolov8n.yaml')  # build a new model from YAML

# Train the model
results = model.train(data='../Dataset/data.yaml', epochs=300, imgsz=352, patience=0)
