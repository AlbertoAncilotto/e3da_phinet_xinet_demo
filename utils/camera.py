import cv2
try:
    from picamera2 import Picamera2
    auto_library = 'picamera'
except:
    auto_library = 'cv2'
    print('Picamera2 not available')

class Camera:
    def __init__(self, width, height, unzoom=2):
        self.width = width
        self.height = height
        self.library = auto_library

        if self.library == 'cv2':
            self.cap = cv2.VideoCapture(0)
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        elif self.library == 'picamera':
            self.picam2 = Picamera2()
            self.picam2.configure(self.picam2.create_preview_configuration(main={"format": 'RGB888', "size": (height*unzoom, width*unzoom)}))
            self.picam2.start()
            # print('TODO: TEST IF RGB OR BGR WITH PICAMERA')

    

    def get_frame(self):
        if self.library == 'cv2':
            ret, frame = self.cap.read()
            if not ret:
                raise Exception("Could not read frame from camera")
            # frame = cv2.resize(frame, (self.width, self.height))
            h,w,c = frame.shape
            ratio = self.height/h
            frame = cv2.resize(frame, (int(w*ratio), self.height))
            frame = frame[:, int(w*ratio/2-self.width/2):int(w*ratio/2+self.width/2)]
            frame = cv2.flip(frame, 1)
            return frame
        elif self.library == 'picamera':
            try:
                frame = self.picam2.capture_array()
                frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.resize(frame, (self.width, self.height))
                frame = cv2.flip(frame, 2)
                return frame
            except Exception as e:
                print(e)
                return None