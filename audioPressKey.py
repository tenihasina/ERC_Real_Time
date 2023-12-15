import speech_recognition as sr
from time import sleep
import keyboard  # pip install keyboard

go = 1


def save_audio(file_name, raw_data):
    with open(file_name, "wb") as f:
        f.write(raw_data)
        f.close()


def quit():
    global go
    print("q pressed, exiting...")
    go = 0
    # print(f"go from quit : {go}")


# press q to quit

def listen():
    r = sr.Recognizer()
    print(sr.Microphone.list_microphone_names())

    mic = sr.Microphone(device_index=8)
    while go:
        try:
            # print(f"go : {go}")
            sleep(0.01)
            with mic as source:
                print("Listening")
                audio = r.listen(source)
                print(r.recognize_google(audio, language="fr-FR"))
                save_audio("test.wav", audio.get_wav_data())

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print("Could not request results from Google Speech Recognition service; {0}".format(e))


if __name__ == '__main__':
    keyboard.on_press_key("q", lambda _: quit())
    listen()
    print("next listen")
    go = 1
    listen()
    print("next listen")
    go = 1
    listen()
