import os

import argparse
import speech_recognition as sr

from src.AVprocessing.utils import save_audio


def record_voice(name, id_micro):
    r = sr.Recognizer()
    original_file = os.path.join("original", f"original_{name}.wav")
    if not (os.path.exists(original_file)):
        with sr.Microphone(device_index=id_micro) as source:
            print("Record new original voice")
            r.adjust_for_ambient_noise(source)
            audio = r.listen(source)
            save_audio(original_file, audio.get_wav_data())
            print("SAVED !")
    else:
        print("Already saved")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", default="UNKNOWN", help="participant")
    parser.add_argument("--id_micro", default=10, help="audio micro")
    args = parser.parse_args()
    record_voice(args.participant, int(args.id_micro))

