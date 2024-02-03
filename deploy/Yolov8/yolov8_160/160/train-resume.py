from ultralytics import YOLO

# Load a model
model = YOLO('runs/detect/train2/weights/last.pt')  # build a new model from YAML

# Train the model
results = model.train(resume=True)


