from ultralytics import YOLO

# Load a model
model = YOLO('ocr.pt')  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data='data.yaml', epochs=100, imgsz=640)
