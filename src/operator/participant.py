import numpy as np
import src.AVprocessing.settings as settings
emodict = {'anger': "anger", 'disgust': "disgust", 'fear': "fear", 'joy': "joy", 'neutral': "neutral", 'sad': "sad", 'surprise': 'surprise', 'sadness':"sadness"}
sentidict = {'positive': ["joy"], 'negative': ["sad", "anger", "disgust", "fear", "sadness"],
             'neutral': ["neutral", "surprise"]}


def get_sentiment(emotion):
    assert emotion in emodict.values()
    try:
        for sentiment, emotions in sentidict.items():
            if emotion in emotions:
                return sentiment
    except AssertionError:
        print("Unknown emotion")


class Participant:

    def __init__(self, name="UNKNOWN"):
        self.name = name
        self.current_attitude = "neutral"
        self.attitudes = []

    def get_major_attitude(self):
        try:
            return max(set(self.attitudes), key=self.attitudes.count)
        except IndexError:
            return None

    def add_attitude(self, name, attitude):
        # assert name in [settings.PARTICIPANTS]
        try:
            success = 0
            if name == self.name:
                self.attitudes.append(attitude)
                success = 1
            else:
                print(f"not for {self.name}")
            return success
        except Exception as e:
            print(e)

    def show_stats(self):
        print(f"{self.name} : CURRENT ATTITUDE {self.current_attitude}")
        major_emotion = self.get_major_attitude()
        print(f"MAJORITY : {major_emotion}, SENTIMENT FROM MAJORITY : {get_sentiment(major_emotion)}")
