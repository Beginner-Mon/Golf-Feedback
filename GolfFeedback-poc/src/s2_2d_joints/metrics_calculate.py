# swing_metrics.py
import math
# ============================================================
# COCO KEYPOINT INDEX MAP (17 Keypoints)
#
#   ID   Name
#   -----------------------------
#   0    Nose
#   1    Left Eye
#   2    Right Eye
#   3    Left Ear
#   4    Right Ear
#   5    Left Shoulder
#   6    Right Shoulder
#   7    Left Elbow
#   8    Right Elbow
#   9    Left Wrist
#   10   Right Wrist
#   11   Left Hip
#   12   Right Hip
#   13   Left Knee
#   14   Right Knee
#   15   Left Ankle
#   16   Right Ankle
#
# Coordinates format:
#   k[i] = (x, y)
#
# Missing pelvis point:
#   Pelvis is assumed as midpoint of left hip (11) and right hip (12)
# ============================================================

def distance(p1, p2):
    """Euclidean distance between two points."""
    return math.sqrt((p2[0]-p1[0])**2 + (p2[1]-p1[1])**2)

def angle_between(a, b, c):
    """Angle at point b formed by points a-b-c in degrees."""
    ba = (a[0]-b[0], a[1]-b[1])
    bc = (c[0]-b[0], c[1]-b[1])
    dot = ba[0]*bc[0] + ba[1]*bc[1]
    mag_ba = math.sqrt(ba[0]**2 + ba[1]**2)
    mag_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    if mag_ba*mag_bc == 0:
        return 0.0
    cos_angle = max(min(dot / (mag_ba*mag_bc), 1.0), -1.0)
    return math.degrees(math.acos(cos_angle))

def atan2_deg(p1, p2):
    """Angle in degrees between two points (horizontal reference)."""
    return math.degrees(math.atan2(p2[1]-p1[1], p2[0]-p1[0]))

def stride_length(keypoints):
    """Distance between left and right ankle."""
    return distance(keypoints[15], keypoints[16])

# -------------------- 15 metric functions --------------------

def shoulder_angle(keypoints):
    return atan2_deg(keypoints[5], keypoints[6])

def upper_tilt(keypoints):
    shoulder_mid = ((keypoints[5][0]+keypoints[6][0])/2, (keypoints[5][1]+keypoints[6][1])/2)
    hip_mid = ((keypoints[11][0]+keypoints[12][0])/2, (keypoints[11][1]+keypoints[12][1])/2)
    ankle_mid = ((keypoints[15][0]+keypoints[16][0])/2, (keypoints[15][1]+keypoints[16][1])/2)

    upper = distance(shoulder_mid, hip_mid)
    lower = distance(hip_mid, ankle_mid)

    return upper / lower

def stance_ratio(keypoints):
    return distance(keypoints[5], keypoints[6]) / stride_length(keypoints)

def head_loc(keypoints, initial_keypoints=None):
    if initial_keypoints:
        return (keypoints[0][0] - initial_keypoints[0][0]) 
    else:
        raise ValueError("initial_keypoints required for head_loc calculation")
    

def shoulder_loc(keypoints):    
    return (keypoints[5][0] - keypoints[15][0]) / max(stride_length(keypoints), 1e-6)

def left_arm_angle(keypoints):
    return angle_between(keypoints[5], keypoints[7], keypoints[9])

def right_arm_angle(keypoints):
    return angle_between(keypoints[6], keypoints[8], keypoints[10])

def hip_rotation(keypoints, initial_keypoints=None):
    angle = atan2_deg(keypoints[11], keypoints[12])
    if initial_keypoints:
        angle -= atan2_deg(initial_keypoints[11], initial_keypoints[12])
    return angle

def hip_shifted(keypoints, addr=None):
    if addr is None:
        raise ValueError("addr (initial keypoints) required for hip_shifted calculation")
    cur_mid = ((keypoints[11][0]+keypoints[12][0])/2, (keypoints[11][1]+keypoints[12][1])/2)
    addr_mid = ((addr[11][0]+addr[12][0])/2, (addr[11][1]+addr[12][1])/2)
    return distance(cur_mid, addr_mid)

def right_leg_angle(keypoints):
    return angle_between(keypoints[12], keypoints[14], keypoints[16])

def shoulder_hanging_back(keypoints):
    return distance(keypoints[15], keypoints[5]) / stride_length(keypoints)

def hip_hanging_back(keypoints):
    return distance(keypoints[15], keypoints[11]) / stride_length(keypoints)

def right_armpit_angle(keypoints):
    pelvis_mid = ((keypoints[11][0]+keypoints[12][0])/2, (keypoints[11][1]+keypoints[12][1])/2)
    return angle_between(keypoints[8], keypoints[6], pelvis_mid)

def weight_shift(keypoints):
    return atan2_deg(keypoints[15], keypoints[11])

def finish_angle(keypoints):
    return atan2_deg(keypoints[15], keypoints[12])

# angle of middle position of shoulder to middle position of hip (Spine) to horizontal line
def spine_angle(keypoints):
    shoulder_mid = ((keypoints[5][0]+keypoints[6][0])/2, (keypoints[5][1]+keypoints[6][1])/2)
    hip_mid = ((keypoints[11][0]+keypoints[12][0])/2, (keypoints[11][1]+keypoints[12][1])/2)
    return atan2_deg(hip_mid, shoulder_mid)

def lower_angle(keypoints):
    return angle_between(keypoints[12], keypoints[14], keypoints[16])

def hip_line(keypoints, addr):
    return ((keypoints[11][0]+keypoints[12][0])/2) - ((addr[11][0]+addr[12][0])/2)

def hip_angle(keypoints, addr):
    return hip_rotation(keypoints, addr)

def right_distance(kp):
    torso_mid = ((kp[5][0]+kp[6][0])/2, (kp[5][1]+kp[6][1])/2)
    return distance(kp[8], torso_mid)


def left_leg_angle(keypoints):
    return angle_between(keypoints[11], keypoints[13], keypoints[15])

# -------------------- Combined calculation --------------------

def calculate_all_metrics(kp, addr=None):
    metrics = {
        "SHOULDER-ANGLE": shoulder_angle(kp),
        "UPPER-TILT": upper_tilt(kp),
        "STANCE-RATIO": stance_ratio(kp),
        "HEAD-LOC": head_loc(kp, addr) if addr else 0,
        "SHOULDER-LOC": shoulder_loc(kp),
        "LEFT-ARM-ANGLE": left_arm_angle(kp),
        "RIGHT-ARM-ANGLE": right_arm_angle(kp),
        "HIP-ROTATION": hip_rotation(kp, addr) if addr else 0,
        "HIP-SHIFTED": hip_shifted(kp, addr) if addr else 0,
        "RIGHT-LEG-ANGLE": right_leg_angle(kp),
        "SHOULDER-HANGING-BACK": shoulder_hanging_back(kp),
        "HIP-HANGING-BACK": hip_hanging_back(kp),
        "RIGHT-ARMPIT-ANGLE": right_armpit_angle(kp),
        "WEIGHT-SHIFT": weight_shift(kp),
        "FINISH-ANGLE": finish_angle(kp),
        "SPINE-ANGLE": spine_angle(kp),
        "LOWER-ANGLE": lower_angle(kp),
        "HIP-LINE": hip_line(kp, addr) if addr else 0,
        "HIP-ANGLE": hip_angle(kp, addr) if addr else 0,
        "RIGHT-DISTANCE": right_distance(kp),
        "LEFT-LEG-ANGLE": left_leg_angle(kp),
    }
    return metrics

# -------------------- Example usage --------------------
if __name__ == "__main__":
    # Fake COCO keypoints for testing: [(x0,y0), (x1,y1), ..., (x16,y16)]
    keypoints = [(0,0)]*17
    metrics = calculate_all_metrics(keypoints)
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")
