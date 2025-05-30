import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import pandas as pd
import datetime
from threading import Thread

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
ds_factor = 0.6

# Load model
emotion_model = Sequential()
emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))
emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax'))
emotion_model.load_weights('model.h5')

cv2.ocl.setUseOpenCL(False)

# Emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Song CSV file paths
music_dist = {
    0: "songs/angry.csv",
    1: "songs/disgusted.csv ",
    2: "songs/fearful.csv",
    3: "songs/happy.csv",
    4: "songs/neutral.csv",
    5: "songs/sad.csv",
    6: "songs/surprised.csv"
}

# ✅ YouTube video IDs per emotion
youtube_links = {
    0: "ZbZSe6N_BXs",  # Angry
    1: "2vjPBrBU-TM",  # Disgusted
    2: "JGwWNGJdvx8",  # Fearful
    3: "y6Sxv-sUYtM",  # Happy
    4: "hT_nvWreIhg",  # Neutral
    5: "RB-RcX5DS5A",  # Sad
    6: "3JZ_D3ELwOQ"   # Surprised
}

global last_frame1
last_frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
global cap1
show_text = [0]

class WebcamVideoStream:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src, cv2.CAP_DSHOW)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True

class VideoCamera(object):
    def get_frame(self):
        global cap1
        global df1
        cap1 = WebcamVideoStream(src=0).start()
        image = cap1.read()
        image = cv2.resize(image, (600, 500))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)
        df1 = pd.read_csv(music_dist[show_text[0]])
        df1 = df1[['Name', 'Album', 'Artist']]
        df1 = df1.head(15)

        for (x, y, w, h) in face_rects:
            cv2.rectangle(image, (x, y - 50), (x + w, y + h + 10), (0, 255, 0), 2)
            roi_gray_frame = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)
            prediction = emotion_model.predict(cropped_img)

            maxindex = int(np.argmax(prediction))
            show_text[0] = maxindex
            cv2.putText(image, emotion_dict[maxindex], (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            df1 = music_rec()

        global last_frame1
        last_frame1 = image.copy()
        img = np.array(Image.fromarray(last_frame1))
        ret, jpeg = cv2.imencode('.jpg', img)
        return jpeg.tobytes(), df1

def music_rec():
    df = pd.read_csv(music_dist[show_text[0]])
    df = df[['Name', 'Album', 'Artist']]
    df = df.head(15)
    return df

# ✅ Correct function to return YouTube video ID for current emotion
def get_current_video():
    return youtube_links[show_text[0]]
