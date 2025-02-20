import asyncio
import cv2
import telegram
from ultralytics import YOLO
import datetime
import threading
import requests
from PIL import Image
import numpy

# Load the exported RKNN model
model = YOLO("./ok_rknn_model")
my_token = "TOKEN HERE"
bot = telegram.Bot(token=my_token)


def get_chat_id():
    url = f"https://api.telegram.org/bot{my_token}/getUpdates"
    print(requests.get(url).json())

def alert(img):
    global last_alert
    cv2.putText(img, "ALARM!!!!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    # New thread to send telegram after 15 seconds
    if (last_alert is None) or (
            (datetime.datetime.utcnow() - last_alert).total_seconds() > 5):
        last_alert = datetime.datetime.utcnow()
        cv2.imwrite("alert.png", cv2.resize(img, dsize=None, fx=0.2, fy=0.2))
        thread = threading.Thread(target=send_telegram)
        thread.start()
    return

def send_telegram():
    try:
        asyncio.run(bot.sendPhoto(chat_id=1246123900, photo=open("alert.png", "rb"), caption="Có chay",))
    except Exception as ex:
        print("Can not send message telegram ", ex)

    print("Send sucess")

# establish and open webcam feed
last_alert =  datetime.datetime.utcnow()

cap = cv2.VideoCapture("Sequence 01_1.mp4")

if not cap.isOpened():
    print("Cannot open camera")
    exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Cannot read camera")
        exit(2)

    # pass frame through model
    frame_resized = cv2.resize(frame, (800, 640))
    # img_bgr = frame_resized
    r = model(frame_resized)[0]

    detects = r.to_json()
    if len(detects)>0:

        # Plot results image
        frame_resized = r.plot()
        alert(frame_resized)

    cv2.imshow('Stream', frame_resized)
    # Break loop on 'q' for quit
    if cv2.waitKey(1) == ord('q'):
        break
