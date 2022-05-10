import os
import time

import deepl
import torch
import socket
from deepl import translator
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer
import speech_recognition as sr

from ERC_dataset import MELD_loader
from model import ERC_model
from utils import make_batch_roberta

import warnings

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
#
# sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
# sock.connect(address)
# address = ('172.18.37.43', 9999)


def prediction():
    start = time.time()
    test_dataset = MELD_loader(test_path, dataclass)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2,
                                 collate_fn=make_batch_roberta)
    last = getCount(test_dataloader)
    time_getcount = time.time() - start
    # start2 = time.time()
    # print(f"get count : {time_getcount} sec")
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


def erc_speech(participant):
    r = sr.Recognizer()
    success = False
    with sr.Microphone() as source:
        print("Say something !")
        audio = r.listen(source)
    try:
        sentence = r.recognize_google(audio, language="fr-FR")
        print("Google Speech Recognition thinks you said " + sentence)
        valid = input('Do you want to send this message ? (y/n)')
        if valid == 'y':
            print("I would have sent that message")
            # sock.send(sentence.encode('utf_8'))
            msg = translator.translate_text(sentence, source_lang="FR", target_lang="EN-US")
            conversation = open(test_path, 'a')
            conversation.write(f"{participant};{msg};neutral;neutral\n")
            success = True
        return success
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))


def erc_keyboard():
    participant = input("Name? ")
    return participant
    # msg = input("Message ")
    msg = translator.translate_text(msg, source_lang="FR", target_lang="EN-US")
    conversation = open(test_path, 'a')
    # conversation.write(f"{participant};{msg};neutral;neutral\n")


def main():
    while 1:
        if erc_speech(erc_keyboard()):
            pred = prediction()
            # sock.send(pred.encode('utf-8'))
            print(f"Prediction : {pred}")


if __name__ == '__main__':
    main()
