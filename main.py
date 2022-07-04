import argparse
import random
import time

import deepl
import torch
from deepl import translator
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import speech_recognition as sr
from speechbrain.pretrained import SpeakerRecognition, foreign_class

from src.AVprocessing import video_recorder as vr
from src.ERC_dataset import MELD_loader
from src.ERC_model import ERC_model
from src.ERC_utils import make_batch_roberta
from src.AVprocessing.settings import *
from src.AVprocessing.utils import *

import warnings

from src.operator.participant import Participant, get_sentiment

warnings.filterwarnings('ignore', category=FutureWarning)
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = ERC_model(model_type, clsNum, False, freeze_type, "pretrained")
# torch.cuda.empty_cache()
# model = model.cuda()
model.load_state_dict(torch.load(modelfile, map_location=torch.device('cpu')))
model.eval()
DEEPL_AUTH_KEY = "ccda91ba-5077-8f1c-3fe5-a0f2a3de6750"
"""Create a Translator object providing your DeepL API authentication key.
To avoid writing your key in source code, you can set it in an environment
variable DEEPL_AUTH_KEY, then read the variable in your Python code:"""
translator = deepl.Translator(DEEPL_AUTH_KEY)

classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                           pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
# Create classifier for Speaker recognition

r = sr.Recognizer()
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(address)
# address = ('172.18.37.43', 9999)
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb")


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
        # return name.split(".")[0]
    return name, pred


def prediction(save_file):
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
    return emotion[pred_logits.argmax(1)]


def getCount(test_dataloader):
    cnt = 0
    for i, data in enumerate(test_dataloader):
        cnt = i
    return cnt


def get_speech(participant, count, id_camera, save_file): #save_file

    name_audio = f"{path_audio}_{participant}_CAM_{id_camera}__{count}.wav"
    try:
        # settings.done_recording = False
        # thread = threading.Thread(target = run, args =(lambda : settings.done_recording, id_camera))
        with sr.Microphone() as source:
            print("Say something !")
            # cameras = start_cameras(id_camera)
            # thread.start()
            name_video = f"{path_video}_CAM_{id_camera}_{participant}_{count}.avi"
            vr.start_video_recording(id_camera=int(id_camera), filename=name_video)
            audio = r.listen(source)
        # settings.done_recording = True
        # thread.join()
        save_audio(name_audio, audio.get_wav_data())
        vr.stop_AVrecording()
        vr.file_manager()
        # kill_cameras(cameras)
        real_name, same_participant = verify_speaker(name_audio, participant)
        if same_participant:
            sentence = r.recognize_google(audio, language="fr-FR")
            out_prob, score, index, text_lab = classifier.classify_file(name_audio)

            save_audio(f"{path_audio}_{real_name}_CAM_{id_camera}_{count}.wav", audio.get_wav_data())
            os.rename(name_video, f"{path_video}_CAM_{id_camera}_{real_name}_{count}.avi")
            if os.path.exists(name_audio):
                os.remove(name_audio)
            text_lab = text_lab[0]
            print(f"Google Speech Recognition thinks {real_name} said : {sentence}")
            # print(f"emotion : {text_lab}, confidence : {score}")

            msg = translator.translate_text(sentence, source_lang="FR", target_lang="EN-US")
            print(f"switch emo : {switch_emo(text_lab)}")
            sentiment = get_sentiment(switch_emo(text_lab))

            text_prediction = f"{real_name};{msg};{switch_emo(text_lab)};{sentiment}\n"
            save_file = addPrediction(save_file, text_prediction)
        # conversation = open(test_path, 'a')
        # conversation.write(f"{real_name};{msg};{switch_emo(text_lab)};{switch_emo(text_lab)}\n")
        else:
            print("not the same speaker, don't do anything")
        return same_participant, save_file, real_name
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", default=0, help="camera id")
    parser.add_argument("--participant", default="UNKNOWN", help="participant's name")
    args = parser.parse_args()
    # cleanFiles()
    create_save_file(settings.test_path)
    count = 0
    save_file = settings.test_path
    while 1:
        success, save_file, real_name = get_speech(args.participant, count, args.id, save_file)
        if success:
            pred = prediction(save_file)
            participant = Participant(name=real_name)
            participant.attitudes.append(pred)
            # sock.send(pred.encode('utf-8'))
            print(f"{participant.name} : current attitude {participant.current_attitude}")
            major_emotion = participant.get_major_attitude()
            print(f"Majority : {major_emotion}, sentiment from majority : {get_sentiment(major_emotion)}")
            count = count + 1


if __name__ == '__main__':
    random.seed(42)
    main()
