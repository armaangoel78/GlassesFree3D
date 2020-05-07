"""Demo code shows how to estimate human head pose.
Currently, human face is detected by a detector from an OpenCV DNN module.
Then the face box is modified a little to suits the need of landmark
detection. The facial landmark detection is done by a custom Convolutional
Neural Network trained with TensorFlow. After that, head pose is estimated
by solving a PnP problem.
"""
from argparse import ArgumentParser
from multiprocessing import Process, Queue

import cv2
import numpy as np

from mark_detector import MarkDetector
from os_detector import detect_os
from pose_estimator import PoseEstimator
from stabilizer import Stabilizer

print("OpenCV version: {}".format(cv2.__version__))

# multiprocessing may not work on Windows and macOS, check OS for safety.
detect_os()

CNN_INPUT_SIZE = 128

# Take arguments from user input.
parser = ArgumentParser()
parser.add_argument("--video", type=str, default=None,
                    help="Video file to be processed.")
parser.add_argument("--cam", type=int, default=None,
                    help="The webcam index.")
args = parser.parse_args()


def get_face(detector, img_queue, box_queue):
    """Get face from image queue. This function is used for multiprocessing"""
    while True:
        image = img_queue.get()
        box = detector.extract_cnn_facebox(image)
        box_queue.put(box)

def init():
    """MAIN"""
    # Video source from webcam or video file.
    video_src = args.cam if args.cam is not None else args.video
    if video_src is None:
        print("Warning: video source not assigned, default webcam will be used.")
        video_src = 0

    cap = cv2.VideoCapture(video_src)
    if video_src == 0:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    _, sample_frame = cap.read()

    # Introduce mark_detector to detect landmarks.
    mark_detector = MarkDetector()

    # Setup process and queues for multiprocessing.
    img_queue = Queue()
    box_queue = Queue()
    img_queue.put(sample_frame)
    box_process = Process(target=get_face, args=(
        mark_detector, img_queue, box_queue,))
    box_process.start()

    # Introduce pose estimator to solve pose. Get one frame to setup the
    # estimator according to the image size.
    height, width = sample_frame.shape[:2]
    pose_estimator = PoseEstimator(img_size=(height, width))

    # Introduce scalar stabilizers for pose.
    pose_stabilizers = [Stabilizer(
        state_num=2,
        measure_num=1,
        cov_process=0.1,
        cov_measure=0.1) for _ in range(6)]

    tm = cv2.TickMeter()

    return cap, video_src, img_queue, box_queue, tm, mark_detector, pose_estimator

def get_pose(cap, video_src, img_queue, box_queue, tm, mark_detector, pose_estimator):
    # Read frame, crop it, flip it, suits your needs.
    frame_got, frame = cap.read()
    if frame_got is False:
        return

    # Crop it if frame is larger than expected.
    # frame = frame[0:480, 300:940]

    # If frame comes from webcam, flip it so it looks like a mirror.
    if video_src == 0:
        frame = cv2.flip(frame, 2)

    # Pose estimation by 3 steps:
    # 1. detect face;
    # 2. detect landmarks;
    # 3. estimate pose

    # Feed frame to image queue.
    img_queue.put(frame)

    # Get face from box queue.
    facebox = box_queue.get()

    if facebox is not None:
        # Detect landmarks from image of 128x128.
        face_img = frame[facebox[1]: facebox[3],
                            facebox[0]: facebox[2]]
        face_img = cv2.resize(face_img, (CNN_INPUT_SIZE, CNN_INPUT_SIZE))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        tm.start()
        marks = mark_detector.detect_marks([face_img])
        tm.stop()

        # Convert the marks locations from local CNN to global image.
        marks *= (facebox[2] - facebox[0])
        marks[:, 0] += facebox[0]
        marks[:, 1] += facebox[1]

        return pose_estimator.solve_pose_by_68_points(marks)

