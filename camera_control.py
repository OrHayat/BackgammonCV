import cv2

# This should be determined according to camera resolution
IMAGE_SCALE_PARAM = 0.5

def init_camera():

    capture = cv2.VideoCapture(0)
    if not capture.isOpened():
        print('Unable to access camera!')
        exit(0)

    return capture

def take_image(video_capture:cv2.VideoCapture):

    ret, frame = video_capture.read()
    ret_img = cv2.resize(frame, (0,0), fx=IMAGE_SCALE_PARAM, fy=IMAGE_SCALE_PARAM)

    return ret_img


def release_camera(video_capture:cv2.VideoCapture):

    video_capture.release()
    return 0
