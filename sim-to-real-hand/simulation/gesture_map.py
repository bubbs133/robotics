GESTURE_MAP = {
    1: "thumbs up",
    2: "extension of index and middle, flexion on others",
    3: "flexion on ring and little finger, extension of others",
    6: "fingers flexed into fist",
    7: "pointing index",
}

# Order matches the actuator order in world.xml:
# [wrist, thumb, index, middle, ring, pinky]
#
# Joint ranges (radians):
#   wrist_joint  : -1.5 (flex) .. 0.2 (extend)
#   thumb_joint  : -1.5 (flex) .. 0.2 (extend)
#   finger joints:  0.0 (extend/straight) .. 1.57 (flex/curled)
#
# So for thumb/wrist, LOWER = more flexed, HIGHER = more extended.
# For the four fingers, HIGHER = more flexed, LOWER = more extended.

WRIST_NEUTRAL = 0.0

THUMB_EXTENDED = 0.2
THUMB_FLEXED = -1.2
THUMB_PARTIAL = -0.5

FINGER_EXTENDED = 0.0
FINGER_FLEXED = 1.45

GESTURE_TARGETS = {
    # thumb up, other four fingers curled into the palm
    1: [WRIST_NEUTRAL, THUMB_EXTENDED, FINGER_FLEXED, FINGER_FLEXED, FINGER_FLEXED, FINGER_FLEXED],

    # index + middle extended (peace-sign-like), thumb/ring/pinky flexed
    2: [WRIST_NEUTRAL, THUMB_FLEXED, FINGER_EXTENDED, FINGER_EXTENDED, FINGER_FLEXED, FINGER_FLEXED],

    # ring + pinky flexed, thumb/index/middle extended
    3: [WRIST_NEUTRAL, THUMB_EXTENDED, FINGER_EXTENDED, FINGER_EXTENDED, FINGER_FLEXED, FINGER_FLEXED],

    # closed fist, thumb tucked across the fingers
    6: [WRIST_NEUTRAL, -1.0, 1.5, 1.5, 1.5, 1.5],

    # index extended straight out, everything else flexed
    7: [WRIST_NEUTRAL, THUMB_PARTIAL, FINGER_EXTENDED, FINGER_FLEXED, FINGER_FLEXED, FINGER_FLEXED],
}