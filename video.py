# import required libraries
import argparse
from vidgear.gears import CamGear
from vidgear.gears import WriteGear
import cv2
import os
from src.AVprocessing.settings import *
from src.AVprocessing.utils import get_local_time_as_str

def main(path, count):
    # define and start the stream on first source ( For e.g #0 index device)
    stream1 = CamGear(source=2, logging=True).start()

    # define and start the stream on second source ( For e.g #1 index device)
    stream2 = CamGear(source=4, logging=True).start()

    # define and start the stream on second source ( For e.g #1 index device)
    stream3 = CamGear(source=6, logging=True).start()

    # Define WriteGear Object with suitable output filename for e.g. `Output.mp4`
    v1 = os.path.join(path, f"Participant1_{count}_{get_local_time_as_str()}.avi")
    v2 = os.path.join(path, f"Participant2_{count}_{get_local_time_as_str()}.avi")
    v3 = os.path.join(path, f"Participant3_{count}_{get_local_time_as_str()}.avi")

    writer1 = WriteGear(output_filename=v1)
    writer2 = WriteGear(output_filename=v2)
    writer3 = WriteGear(output_filename=v3)
    # infinite loop
    while True:

        frameA = stream1.read()
        # read frames from stream1

        frameB = stream2.read()
        # read frames from stream2

        frameC = stream3.read()
        # read frames from stream2

        # check if any frame is None
        if frameA is None or frameB is None or frameC is None:
            # if True break the infinite loop
            break

        writer1.write(frameA)
        writer2.write(frameB)
        writer3.write(frameC)

        key = cv2.waitKey(1) & 0xFF
        # check for 'q' key-press
        if key == ord("q"):
            # if 'q' key-pressed break out
            break

    cv2.destroyAllWindows()
    # close output window

    # safely close video streams
    stream1.stop()
    stream2.stop()
    stream3.stop()

    writer1.close()
    writer2.close()
    writer3.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default=path_video, help="camera id")
    parser.add_argument("--count", default="0", help="participant's name")
    parser.add_argument("--nb_participant", default=3, help="number of participants")
    args = parser.parse_args()
    main(args.path, args.count)
    