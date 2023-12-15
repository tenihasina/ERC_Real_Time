import argparse
import random
import time
# import keyboard
import logging
import logging.config
import deepl
import torch
from deepl import translator
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import speech_recognition as sr
from speechbrain.pretrained import SpeakerRecognition, foreign_class
import subprocess

from src.AVprocessing import video_recorder as vr
from src.ERC_dataset import MELD_loader
from src.ERC_model import ERC_model
from src.ERC_utils import make_batch_roberta
from src.AVprocessing.settings import *
from src.AVprocessing.utils import *

import warnings

from src.operator.participant import Participant, get_sentiment

warnings.filterwarnings('ignore', category=FutureWarning)
# roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
# model = ERC_model(model_type, clsNum, False, freeze_type, "pretrained")
# torch.cuda.empty_cache()
# model = model.cuda()
# model.load_state_dict(torch.load(modelfile, map_location=torch.device('cpu')))
# model.eval()
DEEPL_AUTH_KEY = "ccda91ba-5077-8f1c-3fe5-a0f2a3de6750"
"""Create a Translator object providing your DeepL API authentication key.
To avoid writing your key in source code, you can set it in an environment
variable DEEPL_AUTH_KEY, then read the variable in your Python code:"""
translator = deepl.Translator(DEEPL_AUTH_KEY)

# classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
#                            pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
# Create classifier for Speaker recognition

r = sr.Recognizer()
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(address)
# address = ('172.18.37.43', 9999)
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb")
go = 1

logger_EN = logging.getLogger("transcript_EN")
logger_FR = logging.getLogger("transcript_FR")
log_handler1 = logging.FileHandler("transcriptEN.log")
log_handler2 = logging.FileHandler("transcriptFR.log")
logger_EN.addHandler(log_handler1)
logger_FR.addHandler(log_handler2)


def quit():
    global go
    print("q pressed, exiting...")
    go = 0


def get_mic_from_idCamera(id):
    mic = 10
    if id == 2:
        mic = 8
    elif id == 4:
        mic = 9
    elif id == 6:
        mic = 10
    return mic


def verify_speaker(f1, original):
    name = "UNKNOWN"
    f2 = os.path.join("original/", f"original_{original}.wav")
    score, pred = verification.verify_files(f1, f2)
    if pred:
        print(f"same speaker, confidence : {score}")
        name = original
        print(f"it is {name}")
    else:
        print(f"not the same speaker, confidence : {score}")
    return name, pred


def who_speaks(unknown_speaker):
    speaker = "UNKNOWN"
    success = 0
    for participant in PARTICIPANTS:
        name, same = verify_speaker(unknown_speaker, participant)
        if same:
            speaker = participant
            success = 1
    return speaker, success


def prediction(save_file):
    prediction = "neutral"
    start = time.time()
    test_dataset = MELD_loader(save_file, dataclass)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                                 collate_fn=make_batch_roberta)
    last = getCount(test_dataloader)
    print(last)
    for i, data in enumerate(test_dataloader):
        if last == i:
            b_input, b_label, b_speaker = data
            pred_logits = model(b_input, b_speaker)
            # pred_logits = model(b_input.cuda(), b_speaker)

    # print(b_input.shape)
    print(emotion[pred_logits.argmax(1)])
    print(f"Time prediction for utt {i} : {time.time() - start} sec")
    return prediction


def getCount(test_dataloader):
    cnt = 0
    for i, data in enumerate(test_dataloader):
        cnt = i
    return cnt


def get_transcript_and_audio(participant, count, id_camera):

    name_audio = os.path.join(path_audio, f"{participant}_CAM_{id_camera}_{count}.wav")
    try:
        real_name = "UNKNOWN"
        while go:

            with sr.Microphone(device_index=get_mic_from_idCamera(id_camera)) as source:
                print("Say something !")
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)

            save_audio(name_audio, audio.get_wav_data())

            temp, success = who_speaks(name_audio)
            if success:
                real_name = temp

                sentence = r.recognize_google(audio, language="fr-FR")
                tmp = time.strftime("%d%m%Y_%H_%M_%S", time.localtime())
                save_audio(f"{path_audio}/{tmp}_{real_name}_CAM_{id_camera}_{count}.wav", audio.get_wav_data())

                if os.path.exists(name_audio):
                    os.remove(name_audio)

                print(f"Google Speech Recognition thinks {real_name} said : {sentence}")
                logger_FR.info(f"{real_name} : {sentence}")
                msg = translator.translate_text(sentence, source_lang="FR", target_lang="EN-US")
                logger_EN.info(f"{real_name} : {msg}")
        return 1
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return 1
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))
        return 1


def get_speech(participant, count, id_camera, save_file):  # save_file

    name_audio = os.path.join(path_audio, f"{participant}_CAM_{id_camera}_{count}.wav")
    try:
        real_name = "UNKNOWN"
        while go:
            p = subprocess.Popen(['/home/sandratra/anaconda3/envs/thesis/bin/python3',
                                  '/home/sandratra/Documents/thesis/ERC_Real_Time/video.py',
                                 #  "--path",
                                 # f"{path_video}",
                                  "--count",
                                 f"{count}"])
            with sr.Microphone(device_index=get_mic_from_idCamera(id_camera)) as source:
                print("Say something !")
                r.adjust_for_ambient_noise(source)
                audio = r.listen(source)

            save_audio(name_audio, audio.get_wav_data())
            p.kill()
            # vr.stop_AVrecording()
            # vr.file_manager()

            temp, success = who_speaks(name_audio)
            if success:
                real_name = temp

                sentence = r.recognize_google(audio, language="fr-FR")
                out_prob, score, index, text_lab = classifier.classify_file(name_audio)

                save_audio(f"{path_audio}_{real_name}_CAM_{id_camera}_{count}.wav", audio.get_wav_data())

                if os.path.exists(name_audio):
                    os.remove(name_audio)
                text_lab = text_lab[0]
                print(f"Google Speech Recognition thinks {real_name} said : {sentence}")
                timestamp = time.strftime("%d%m%Y_%H_%M_%S", time.localtime())
                logger_FR.info(f"{timestamp} {real_name} : {sentence}" )
                msg = translator.translate_text(sentence, source_lang="FR", target_lang="EN-US")
                logger_EN.info(f"{timestamp} {real_name} : {msg}")

                print(f"switch emo : {switch_emo(text_lab)}")
                sentiment = get_sentiment(switch_emo(text_lab))

                text_prediction = f"{real_name};{msg};{switch_emo(text_lab)};{sentiment}\n"
                save_file = addPrediction(save_file, text_prediction)

            return success, save_file, real_name
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return success, save_file, real_name
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=2, help="camera id")
    parser.add_argument("--participant", default="UNKNOWN", help="participant's name")
    args = parser.parse_args()
    # cleanFiles()
    # keyboard.on_press_key("q", lambda _: quit())
    create_save_file(settings.test_path)
    count = 0
    save_file = settings.test_path
    # p1 = Participant(name="Jason")
    # p2 = Participant(name="Aldo")
    # p3 = Participant(name="Mathieu")
    # p4 = Participant(name="UNKNOWN")

    while 1:
        # success, save_file, real_name = get_speech(args.participant, count, args.id, save_file)
        success = get_transcript_and_audio(args.participant, count, args.id)
        if success:
            # pred = prediction(save_file)
            # for p in [p1, p2, p3, p4]:
            #     show_pred = p.add_attitude(real_name, pred)
            #     if show_pred:
            #         p.show_stats()
            count = count + 1


if __name__ == '__main__':
    random.seed(42)
    main()
