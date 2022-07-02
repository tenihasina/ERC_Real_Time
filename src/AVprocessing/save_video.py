import cv2
import argparse


# def record_video():
#





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="record/default.mp4", help="video path")
    parser.add_argument("--id", default=0, help="camera id")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.id)

    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = 480
    height = 640
    # writer = cv2.VideoWriter(args.path, cv2.VideoWriter_fourcc(*'DIVX'), 20, (width, height)) #, cv2.VideoWriter_fourcc(*'DIVX')
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    writer = cv2.VideoWriter(f"{args.path}_{args.id}.mp4", fourcc, 20,
                             (width, height))  # , cv2.VideoWriter_fourcc(*'DIVX')
    while True:
        ret, frame = cap.read()

        writer.write(frame)

        cv2.imshow(f"CAMERA {args.id}", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
