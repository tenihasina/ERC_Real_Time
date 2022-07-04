import numpy as np

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
