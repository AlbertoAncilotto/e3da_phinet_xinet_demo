import cv2
import os, sys, platform
import numpy as np
import onnxruntime
onnxruntime.set_default_logger_severity(3)
import time
from utils.align_face import dealign, align_img
from utils.prepare_data import LandmarkModel
from utils import yolo_nms
from utils import camera


backgrounds = {
    0: ["slides_bg/slide_0.png", (0, 0, 1.0)],  # Only background, no frame
    1: ["slides_bg/slide_1.png", (430, 450, 1.7)],  
    2: ["slides_bg/slide_1.png", (430, 450, 1.7)],  
    3: ["slides_bg/slide_2.png", (1300, 450, 1.8)], 
}

# Preload and resize all backgrounds before the loop
background_images = []
for mode, (bg_path, (center_x, center_y, scale)) in backgrounds.items():
    background_img = cv2.imread(bg_path)
    resized_bg = cv2.resize(background_img, (1800, 900))
    background_images.append((resized_bg, (center_x, center_y, scale)))

if platform.machine() in ['AMD64', 'x86_64', 'i386', 'x86', 'i686']:
    IS_EMBEDDED = False
    cv2.namedWindow("Phinet Multi-Mode")
else:
    IS_EMBEDDED = True
    print("Running on embedded device")
    cv2.namedWindow("Phinet Multi-Mode", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Phinet Multi-Mode", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

cv2.imshow("Phinet Multi-Mode", background_images[0][0])
cv2.waitKey(1)

# Load models
object_model = onnxruntime.InferenceSession("phinet.onnx", providers=['CPUExecutionProvider'])  # Object detection model
pose_model = onnxruntime.InferenceSession("xinet-pose.onnx", providers=['CPUExecutionProvider'])  # Pose detection model

# Load Face Swapping components
base_dir = 'demo_images'
landmarkModel = LandmarkModel(name='landmarks')
landmarkModel.prepare(ctx_id=0, det_thresh=0.1, det_size=(128, 128))
inf_sessions = []

for model in os.listdir(base_dir):
    if not model.endswith('onnx'):
        continue
    ort_session = onnxruntime.InferenceSession(os.path.join(base_dir, model), providers=['CPUExecutionProvider'])
    ref_img = cv2.imread(os.path.join(base_dir, model.replace('.onnx', '')))
    inf_sessions.append({'session': ort_session, 'img': ref_img})

def switch_mode(delta):
    global mode, mode_start_time
    mode = (mode + delta) % 4
    mode_start_time = time.time()

def mouse_click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        switch_mode(1)
    if event == cv2.EVENT_RBUTTONDOWN:
        switch_mode(-1)

# Mode settings
mode_durations = {0: 2, 1: 5, 2: 5, 3: 15}
mode = 0  # Start with mode 0
curr_model = 0 # for face swapping
ort_session = inf_sessions[curr_model]['session']
mode_start_time = time.time()

cap = camera.Camera(480, 320)

cv2.setMouseCallback("Phinet Multi-Mode", mouse_click_event)

while True:
    frame = cap.get_frame()
    frame_start = time.time()
    background_img, (center_x, center_y, scale) = background_images[mode]

    # Automatically switch modes after the specified time
    if IS_EMBEDDED and time.time() - mode_start_time > mode_durations[mode]:
        switch_mode(1)

    if mode == 0:  # Only Background
        annotated_frame = background_img

    elif mode == 1:  # Object Detection
        img = cv2.resize(frame, (320, 256))
        input_image = np.expand_dims(img.astype(np.float32).transpose(2, 0, 1) / 255.0, axis=0)
        ort_inputs = {object_model.get_inputs()[0].name: input_image}
        results = object_model.run(None, ort_inputs)[0]
        yolo_nms.plot_boxes(results, img)
        annotated_frame = cv2.resize(img, (480, 320))

    elif mode == 2:  # Pose Detection
        img = cv2.resize(frame, (320, 256))
        input_image = np.expand_dims(img.astype(np.float32).transpose(2, 0, 1) / 255.0, axis=0)
        ort_inputs = {pose_model.get_inputs()[0].name: input_image}
        results = pose_model.run(None, ort_inputs)[0]
        annotated_frame = yolo_nms.post_process_multi(img, results[0], score_threshold=.5)
        annotated_frame = cv2.resize(annotated_frame, (480, 320))
        
    elif mode == 3:  # Face Swapping
        landmark = landmarkModel.get(frame)
        if landmark is not None:
            att_img, back_matrix = align_img(frame, landmark)
            att_img = np.transpose(cv2.cvtColor(att_img, cv2.COLOR_BGR2RGB), (2, 0, 1)).astype(np.float32) / 255.0
            ort_inputs = {ort_session.get_inputs()[0].name: att_img[None, ...]}
            [res, mask] = ort_session.run(None, ort_inputs)
            res = cv2.cvtColor(np.transpose(res[0] * 255, (1, 2, 0)), cv2.COLOR_RGB2BGR)
            mask = np.transpose(mask[0], (1, 2, 0))
            res = dealign(res, frame, back_matrix, mask)
            annotated_frame = res
        else:
            annotated_frame = frame

        # Overlay reference image
        resized_img = cv2.resize(inf_sessions[curr_model]['img'], (int(frame.shape[1] / 4), int(frame.shape[1] / 4)))
        h, w = resized_img.shape[:2]
        annotated_frame[:h, -w:, :] = resized_img

        if time.time() - mode_start_time > mode_durations[mode]:
            curr_model += 1
            curr_model %= len(inf_sessions)
            ort_session = inf_sessions[curr_model]['session']
            mode_start_time = time.time()

    # Paste the current frame onto the background
    if mode != 0:
        if not IS_EMBEDDED:
            resized_frame = cv2.resize(annotated_frame, (0, 0), fx=scale, fy=scale)
            h, w = resized_frame.shape[:2]
            x1, y1 = center_x - w // 2, center_y - h // 2
            x2, y2 = x1 + w, y1 + h
            background_img[y1:y2, x1:x2] = resized_frame
            annotated_frame = background_img

    if IS_EMBEDDED:
        #add text 'energy: {frame_time*5}mJ' o bottom left corner
        cv2.putText(annotated_frame, f'Energy: {(time.time()-frame_start)*4:.2f} J', (10, 310), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow("Phinet Multi-Mode", annotated_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('d'):
        switch_mode(1)
    elif key == ord('a'):
        switch_mode(-1)

cap.release()
cv2.destroyAllWindows()
