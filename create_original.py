import os

import argparse
import speech_recognition as sr

from src.AVprocessing.utils import save_audio


def record_voice(name=""):
    r = sr.Recognizer()
    original_file = os.path.join("original", f"original_{name}.wav")
    if not (os.path.exists(original_file)):
        with sr.Microphone() as source:
            print("Record new original voice")
            audio = r.listen(source)
            save_audio(original_file, audio.get_wav_data())
            print("SAVED !")
    else:
        print("Already saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", default="UNKNOWN", help="participant")
    args = parser.parse_args()
    record_voice(args.participant)

