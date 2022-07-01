import os
import time

import deepl
import torch
from deepl import translator
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import speech_recognition as sr
from speechbrain.pretrained import SpeakerRecognition, foreign_class

from src.ERC_dataset import MELD_loader
from src.ERC_model import ERC_model
from src.ERC_utils import make_batch_roberta, create_save_file

import warnings

import regex as re

original_files = ["record/original_Sassa.wav"]
path_audio = "record/audio"

warnings.filterwarnings('ignore', category=FutureWarning)
roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "joy", 'neutral': "neutral", 'sadness': "sad",
           'surprise': 'surprise'}
emotion = list(emodict.values())
test_path = "test_conversation.txt"

pretrained = 'roberta-large'
cls = 'emotion'
initial = 'pretrained'
dataset = "MELD"
model_type = "roberta-large"
freeze_type = "no_freeze"
dataclass = "emotion"
save_path = os.path.join(dataset + '_models', model_type, initial, freeze_type, dataclass, str(1.0))
modelfile = os.path.join("models", save_path, "model.bin")
clsNum = 7
model = ERC_model(model_type, clsNum, False, freeze_type, "pretrained")
# torch.cuda.empty_cache()
# model = model.cuda()
model.load_state_dict(torch.load(modelfile, map_location=torch.device('cpu')))
model.eval()
DEEPL_AUTH_KEY = "ccda91ba-5077-8f1c-3fe5-a0f2a3de6750"
# Create a Translator object providing your DeepL API authentication key.
# To avoid writing your key in source code, you can set it in an environment
# variable DEEPL_AUTH_KEY, then read the variable in your Python code:
translator = deepl.Translator(DEEPL_AUTH_KEY)
classifier = foreign_class(source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                           pymodule_file="custom_interface.py", classname="CustomEncoderWav2vec2Classifier")
verification = SpeakerRecognition.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb",
                                               savedir="pretrained_models/spkrec-ecapa-voxceleb")


#
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(address)
# address = ('172.18.37.43', 9999)

def switch_emo(emotion):
    emo = 'neutral'
    if emotion == "['ang']":
        emo = 'anger'
    elif emotion == "['sad']":
        emo = 'sadness'
    elif emotion == "['hap']":
        emo = "joy"
    return emo


def save_audio(file_name, raw_data):
    with open(file_name, "wb") as f:
        f.write(raw_data)


def verify_speaker(f1):
    name = "UNkNOWN"
    for f2 in original_files:
        score, pred = verification.verify_files(f1, f2)
        if pred:
            print(f"same speaker, confidence : {score}")
            name = f2.split("_")[1]
        else:
            print(f"not the same speaker : {score}")
        return name.split(".")[0]


def prediction():
    start = time.time()
    test_dataset = MELD_loader(test_path, dataclass)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                                 collate_fn=make_batch_roberta)
    last = getCount(test_dataloader)
    time_getcount = time.time() - start
    # start2 = time.time()
    # print(f"get count : {time_getcount} sec")
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


def erc_speech(participant, count):
    r = sr.Recognizer()
    # success = False
    name_audio = f"{path_audio}_{participant}_{count}.wav"
    try:
        with sr.Microphone() as source:
            print("Say something !")
            audio = r.listen(source)

        save_audio(name_audio, audio.get_wav_data())
        sentence = r.recognize_google(audio, language="fr-FR")
        out_prob, score, index, text_lab = classifier.classify_file(name_audio)
        real_name = verify_speaker(name_audio)
        save_audio(f"{path_audio}_{real_name}_{count}.wav", audio.get_wav_data())
        print(f"Google Speech Recognition thinks {real_name} said : {sentence}")
        print(f"emotion : {text_lab}, confidence : {score}")

        msg = translator.translate_text(sentence, source_lang="FR", target_lang="EN-US")
        conversation = open(test_path, 'a')
        conversation.write(f"{real_name};{msg};{switch_emo(text_lab)};{switch_emo(text_lab)}\n")
        success = True
        return success
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


# def erc_keyboard():
#     participant = input("Name? ")
#     return participant
#     # msg = input("Message ")
#     # msg = translator.translate_text(msg, source_lang="FR", target_lang="EN-US")
#     # conversation = open(test_path, 'a')
#     # conversation.write(f"{participant};{msg};neutral;neutral\n")


def main():
    create_save_file(test_path)
    count = 0
    while 1:
        if erc_speech("Participant", count):
            pred = prediction()
            # sock.send(pred.encode('utf-8'))
            print(f"Prediction : {pred}")
            count = count + 1


if __name__ == '__main__':
    main()
