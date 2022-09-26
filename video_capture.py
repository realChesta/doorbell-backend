from threading import Thread
import cv2

class VideoScreenshot(object):
    def __init__(self, src):
        # Create a VideoCapture object
        self.capture = cv2.VideoCapture(src)
        (self.status, self.frame) = self.capture.read()

        # Start the thread to read frames from the video stream
        self.thread = Thread(target=self.update, args=())
        self.thread.daemon = True
        self.thread.start()

    def release(self):
        self.capture.release()

    def update(self):
        # Read the next frame from the stream in a different thread
        while True:
            if self.capture.isOpened():
                (self.status, self.frame) = self.capture.read()

    def get_frame(self):
        return self.frame
