import os
import re

import cv2
from speechbrain.pretrained import SpeakerRecognition

import src.AVprocessing.settings as settings
from src.ERC_utils import create_save_file


def save_audio(file_name, raw_data):
    with open(file_name, "wb") as f:
        f.write(raw_data)
        f.close()


def switch_emo(t_emo):
    emo = 'neutral'
    if t_emo == "ang":
        emo = 'anger'
    elif t_emo == "sad":
        emo = 'sadness'
    elif t_emo == "hap":
        emo = "joy"
    return emo


def run(stop, id_camera):
    print(f"Starting thread save_video for camera {id_camera}")
    cap = cv2.VideoCapture(id_camera)
    width = 480
    height = 640
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(f"{settings.path_video}_{id_camera}.mp4", fourcc, 20,
                             (width, height))  # , cv2.VideoWriter_fourcc(*'DIVX')

    while True:
        ret, frame = cap.read()
        writer.write(frame)
        cv2.imshow(f"CAMERA {id_camera}", frame)
        if stop():
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


def countLinesWithoutBlank(file_path):
    try:
        count = 0
        if os.path.exists(file_path):
            with open(file_path) as fp:
                for line in fp:
                    if line.strip():
                        count += 1
        return count
    except OSError as e:
        print(e.errno)


def addPrediction(file_path, text):
    try:
        saveFile = file_path
        nbLines = countLinesWithoutBlank(file_path)
        if nbLines > settings.MAX_LINES:
            # create new file
            idFile = int(file_path.split("_")[-1].split(".")[0])
            newFile = f"{file_path.split('_')[0]}_{idFile+1}.txt"
            create_save_file(newFile)
            saveFile = newFile

        with(open(saveFile, 'a')) as f:
            f.write(text)
            f.close()
        return saveFile
    except OSError as e:
        print(e.errno)


def cleanFiles():
    audio = ".*(.wav)"
    video = ".*(.avi)"
    audio_main = "(audio).*.wav"
    print('cleaning working directory')
    for f in os.listdir(os.getcwd()):
        wd_match = re.match(audio_main, f)
        if wd_match:
            os.remove(os.path.join(os.getcwd(),f))
    print('cleaning logs/record')
    for f in os.listdir("logs/record"):
        audio_match = re.match(audio, f)
        video_match = re.match(video, f)
        if audio_match or video_match:
            os.remove(os.path.join("logs/record", f))
    print('DONE CLEANING')
