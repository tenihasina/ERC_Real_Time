import cv2
import threading

def list_ports():

# Test the ports and returns a tuple with the available ports and the ones that are working.

    is_working = True
    dev_port = 0
    working_ports = []
    available_ports = []
    while dev_port <= 10:
        camera = cv2.VideoCapture(dev_port)
        if not camera.isOpened():
            is_working = False
            print("Port %s is not working." %dev_port)
        else:
            is_reading, img = camera.read()
            w = camera.get(3)
            h = camera.get(4)
            if is_reading:
                print("Port %s is working and reads images (%s x %s)" %(dev_port, h, w))
                working_ports.append(dev_port)
            else:
                print("Port %s for camera ( %s x %s) is present but does not reads." %(dev_port, h, w))
                available_ports.append(dev_port)
        dev_port += 1
    return available_ports, working_ports

class camThread(threading.Thread):
    def __init__(self, previewName, camID):
        threading.Thread.__init__(self)
        self.previewName = previewName
        self.camID = camID
    def run(self):
        print("Starting " + self.previewName)
        camPreview(self.previewName, self.camID)

def camPreview(previewName, camID):
    print(f"thread for {previewName} is starting")
    cv2.namedWindow(previewName)
    print(f"named window for {previewName} is starting")
    cam = cv2.VideoCapture(camID)
    print(f"capture video for {previewName} is starting")

    if cam.isOpened():  # try to get the first frame
        rval, frame = cam.read()
    else:
        rval = False

    while rval:
        cv2.imshow(previewName, frame)
        rval, frame = cam.read()
        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
    cv2.destroyWindow(previewName)

list_ports()
