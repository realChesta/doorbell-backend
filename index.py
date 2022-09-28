from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from time import sleep
import cv2
import numpy as np
import webserver
import websockets
import asyncio
from video_capture import VideoScreenshot
import json
import os 

# THE CONFIG MUST HAVE THE FOLLOWING STRUCTURE:
# {
#     "hostname": string,
#     "http-port": number,
#     "ws-port": number,
#     "capture-interval": float,
#     "rtsp-url": string
# }

BRIGHTNESS_THRESHOLD = 100
MIN_ACTIVE_FRAMES = 3

DIR_PATH = os.path.dirname(os.path.realpath(__file__))

CONNECTIONS = set()

with open(os.path.join(DIR_PATH, 'config/settings.json')) as f:
    config = json.load(f)

def start_server(srv):
    print('Starting webserver...')
    thread = threading.Thread(target=srv.serve_forever);
    thread.start();
    print('http running on ' + config['hostname'] + ':' + str(config['http-port']))

def start_video_thread():
    thread = threading.Thread(target=video_main);
    thread.start();


def find_corners(im):
    """ 
    Find "card" corners in a binary image.
    Return a list of points in the following format: [[640, 184], [1002, 409], [211, 625], [589, 940]] 
    The points order is top-left, top-right, bottom-left, bottom-right.
    """

    # Better approach: https://stackoverflow.com/questions/44127342/detect-card-minarea-quadrilateral-from-contour-opencv

    # Find contours in img.
    cnts, hierarchy = cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # [-2] indexing takes return value before last (due to OpenCV compatibility issues).

    # final_contours = []
    # # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for cnt in cnts:
    #     approx = cv2.contourArea(cnt)
    #     # print(approx)
    #     if approx > 20000:
    #         final_contours.append(cnt)
    # final_contours.sort(key=cv2.contourArea)
    # final_contours.pop(0)
    
    # Find the contour with the maximum area (required if there is more than one contour).
    c = max(cnts, key=cv2.contourArea) # final_contours[0]

    # https://stackoverflow.com/questions/41138000/fit-quadrilateral-tetragon-to-a-blob
    epsilon = 0.1*cv2.arcLength(c, True)
    box = cv2.approxPolyDP(c, epsilon, True)

    # Draw box for testing
    tmp_im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(tmp_im, [box], 0, (0, 255, 0), 10)
    # cv2.imshow("tmp_im", tmp_im)
    # show_img(tmp_im)

    box = np.squeeze(box).astype(np.float32)  # Remove redundant dimensions


    # Sorting the points order is top-left, top-right, bottom-right, bottom-left.
    # Note: 
    # The method I am using is a bit of an "overkill".
    # I am not sure if the implementation is correct.
    # You may sort the corners using simple logic - find top left, bottom right, and match the other two points.
    ############################################################################
    # Find the center of the contour
    # https://docs.opencv.org/3.4/dd/d49/tutorial_py_contour_features.html
    M = cv2.moments(c)
    cx = M['m10']/M['m00']
    cy = M['m01']/M['m00']
    center_xy = np.array([cx, cy])

    cbox = box - center_xy  # Subtract the center from each corner

    # For a square the angles of the corners are:
    # -135   -45
    #
    #
    # 135     45
    ang = np.arctan2(cbox[:,1], cbox[:,0]) * 180 / np.pi  # Compute the angles from the center to each corner

    # Sort the corners of box counterclockwise (sort box elements according the order of ang).
    box = box[ang.argsort()]
    ############################################################################

    # Reorder points: top-left, top-right, bottom-left, bottom-right
    coor = np.float32([box[0], box[1], box[3], box[2]])

    return coor

def get_perspective_transform(img, width, height):
    """
    This function takes an image and returns the necessary transformation
    to perspective warp to the small screen.
    """
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.bitwise_not(gray_img)
    thresh_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # cnts, hierarchy = cv2.findContours(thresh_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # final_contours = []
    # # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    # for cnt in cnts:
    #     # print(cnt)
    #     # pass
    #     approx = cv2.contourArea(cnt)
    #     # print(approx)
    #     if approx > 20000:
    #         final_contours.append(cnt)

    orig_im_coor = find_corners(thresh_img)
    new_image_coor =  np.float32([[0, 0], [width, 0], [0, height], [width, height]])

    return cv2.getPerspectiveTransform(orig_im_coor, new_image_coor)

def rotate_image(image, angle):
  image_center = tuple(np.array(image.shape[1::-1]) / 2)
  rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
  result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
  return result

def calculate_brightness(img):
    if len(img.shape) == 3:
        # Colored RGB or BGR (*Do Not* use HSV images with this function)
        # create brightness with euclidean norm
        return np.average(np.linalg.norm(img, axis=2)) / np.sqrt(3)
    else:
        # Grayscale
        return np.average(img)

def show_img(img):
    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()
    cv2.imshow('perspective cam', img)

async def register(websocket):
    CONNECTIONS.add(websocket)
    try:
        async for message in websocket:
            print(message)
    finally:
        CONNECTIONS.remove(websocket)

async def ws_main():
    print("starting ws server...")
    async with websockets.serve(register, config['hostname'], config['ws-port']):
        print("ws started, waiting for connections!")
        await asyncio.Future()
    print("done with ws!")


def video_main():
    #Import image
    print("connecting to the camera...", end=None)
    capture = VideoScreenshot(config['rtsp-url'])
    print("RTSP connected!")

    # ret, frame = vcap.read()
    frame = capture.get_frame()
    width, height = 960, 720
    P = get_perspective_transform(frame, width, height)
    webServer = HTTPServer((config['hostname'], config['http-port']), webserver.CamServer)
    start_server(webServer)
    is_active = False
    current_active_count = 0

    while True:
        frame = capture.get_frame()
        # ret, frame = vcap.read()
        # frame = rotate_image(frame, 90)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # cv2.drawContours(output, final_contours, -1, (0,255,0), 10)
        perspective = cv2.warpPerspective(frame, P, (width, height))
        webserver.CamServer.snapshot = perspective

        brt = calculate_brightness(perspective)
        if not is_active and brt >= BRIGHTNESS_THRESHOLD:
            current_active_count += 1
            if current_active_count >= MIN_ACTIVE_FRAMES:
                is_active = True
                print("motion now active!")
                websockets.broadcast(CONNECTIONS, 'motion-start')
        elif is_active and brt <= BRIGHTNESS_THRESHOLD:
            current_active_count = 0
            is_active = False
            print("motion no longer active!")
            websockets.broadcast(CONNECTIONS, 'motion-end')

        
        sleep(config['capture-interval'])

    # vcap.release()
    capture.release()
    cv2.waitKey(0)

start_video_thread()
asyncio.run(ws_main())
