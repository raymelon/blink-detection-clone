# blink-detection-clone


#### 1. install external dependencies

```bash
sudo apt-get install cmake
```

#### 2. create a python env

```bash
python3 -m venv .venv
```

#### 3. install python dependencies

```bash
pip install scipy imutils numpy dlib opencv-python
```

#### 4. run

```bash
python3 detect_blinks.py --shape-predictor shape_predictor_68_face_landmarks.dat --video blink_detection_demo.mp4
```
