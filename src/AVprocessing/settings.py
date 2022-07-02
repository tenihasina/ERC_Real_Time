import os

original_files = ["record/original_Sassa.wav"]
path_audio = "record/audio"
path_video = "record/video"

emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "joy", 'neutral': "neutral", 'sadness': "sad",
           'surprise': 'surprise'}
emotion = list(emodict.values())

test_path = "../../conversation_0.txt"

pretrained = 'roberta-large'
cls = 'emotion'
initial = 'pretrained'
dataset = "MELD"
model_type = "roberta-large"
freeze_type = "no_freeze"
dataclass = "emotion"
save_path = os.path.join(dataset + '_models', model_type, initial, freeze_type, dataclass, str(1.0))
modelfile = os.path.join("../../models", save_path, "model.bin")
clsNum = 7

id_cameras = [0, 2, 4]
done_recording = False
MAX_LINES = 5
#
