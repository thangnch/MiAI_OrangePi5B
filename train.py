from ultralytics import YOLO
model = YOLO("yolo11n.pt")
output_path="./train.yaml"
train_results = model.train(
    data=output_path,
    epochs=10,  # number of training epochs
    imgsz=480,  # training image size
    device=[0])  # device to run

model.save("ok.pt")