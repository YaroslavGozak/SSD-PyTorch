from ultralytics import YOLO
model = YOLO('yolo26n.pt')  # Auto-downloads if not present

model.save("trained_models/coco-yolo26n/best.pt")