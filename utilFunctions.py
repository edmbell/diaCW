# python utilFunctions.py

import numpy as np
import cv2

def rgbToGray(frame):
    # convert rgb image to grayscale, resize to 96x96, normalize to [0, 1]
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (96, 96))
    return resized / 255.0

def getFrameStack(frameQueue):
    # stack frames into shape (stackNum, height, width) for input to model
    return np.stack(frameQueue, axis=0)

def mapDiscreteAction(actionIndex):
    # maps a discrete action index to a continuous [steer, throttle, brake] vector
    actionMap = {
        0: np.array([0.0, 0.0, 0.0]),   # no action
        1: np.array([0.0, 1.0, 0.0]),   # accelerate forward
        2: np.array([-1.0, 1.0, 0.0]),  # turn left with gas
        3: np.array([1.0, 1.0, 0.0]),   # turn right with gas
        4: np.array([0.0, 0.0, 0.8]),   # brake
    }
    return actionMap.get(actionIndex, np.array([0.0, 0.0, 0.0]))  # fallback to no-op if invalid index
