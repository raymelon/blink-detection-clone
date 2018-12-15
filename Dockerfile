FROM python:3

ADD detect_blinks.py /
ADD shape_predictor_68_face_landmarks.dat /
ADD blink_detection_demo.mp4 /
RUN apt-get -y update
RUN apt-get -y install cmake
RUN pip install scipy imutils numpy dlib opencv-python
CMD [ "python", "detect_blinks.py", "--shape-predictor", "shape_predictor_68_face_landmarks.dat", "--video", "blink_detection_demo.mp4" ]
