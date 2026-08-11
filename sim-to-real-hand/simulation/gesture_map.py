GESTURE_MAP = {
    1: "thumbs up",
    2: "extension of index and middle, flexion on others",
    3: "flexion on ring and little finger, extension of others",
    6: "fingers flexed into fist",
    7: "pointing index",
}

def robot_gesture(gesture_prediction):
    if gesture_prediction == 1:
        print(GESTURE_MAP.keys)