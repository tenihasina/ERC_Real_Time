import speech_recognition as sr
import socket

# Init socket
ip = 'localhost'
port = 10000
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
proxy_address = (ip, port)
# sock.connect(proxy_address)

while 1:
    # obtain audio from the microphone
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        audio = r.listen(source)

    try:
        sentence = r.recognize_google(audio)
        print("Google Speech Recognition thinks you said " + sentence)
        valid = input('Do you want to send this message ? (y/n)')
        if valid == 'y':
            print("I would have sent that message")
            # sock.send(sentence.encode('utf_8'))

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print("Could not request results from Google Speech Recognition service; {0}".format(e))