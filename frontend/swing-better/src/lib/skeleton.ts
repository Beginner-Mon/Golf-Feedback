import * as THREE from 'three';

export const SKELETON_CONFIG = {
    JOINT_LABELS: [
        "hip",          // 0: p1
        "r_hip",        // 1: p2
        "r_knee",       // 2: p3
        "r_ankle",      // 3: p4
        "l_hip",        // 4: p5
        "l_knee",       // 5: p6
        "l_ankle",      // 6: p7
        "spine",        // 7: p8
        "neck",         // 8: p9
        "neck_base",    // 9: p10
        "head",         // 10: p11
        "l_shoulder",   // 11: p12
        "l_elbow",      // 12: p13
        "l_wrist",      // 13: p14
        "r_shoulder",   // 14: p15
        "r_elbow",      // 15: p16
        "r_wrist"       // 16: p17
    ],

    JOINT_COLORS: [
        0x888888, // 0: hip (center)
        0xff0000, // 1: r_hip (right)
        0xff0000, // 2: r_knee (right)
        0xff0000, // 3: r_ankle (right)
        0x0000ff, // 4: l_hip (left)
        0x0000ff, // 5: l_knee (left)
        0x0000ff, // 6: l_ankle (left)
        0x888888, // 7: spine (center)
        0x888888, // 8: neck (center)
        0x888888, // 9: neck_base (center)
        0x888888, // 10: head (center)
        0x0000ff, // 11: l_shoulder (left)
        0x0000ff, // 12: l_elbow (left)
        0x0000ff, // 13: l_wrist (left)
        0xff0000, // 14: r_shoulder (right)
        0xff0000, // 15: r_elbow (right)
        0xff0000  // 16: r_wrist (right)
    ],

    BONE_CONNECTIONS: [
        // Legs
        [0, 1],   // hip -> r_hip
        [1, 2],   // r_hip -> r_knee
        [2, 3],   // r_knee -> r_ankle
        [0, 4],   // hip -> l_hip
        [4, 5],   // l_hip -> l_knee
        [5, 6],   // l_knee -> l_ankle

        // Spine
        [0, 7],   // hip -> spine
        [7, 8],   // spine -> neck
        [8, 9],   // neck -> neck_base
        [9, 10],  // neck_base -> head

        // Left arm
        [8, 11],  // neck -> l_shoulder
        [11, 12], // l_shoulder -> l_elbow
        [12, 13], // l_elbow -> l_wrist

        // Right arm
        [8, 14],  // neck -> r_shoulder
        [14, 15], // r_shoulder -> r_elbow
        [15, 16], // r_elbow -> r_wrist
    ],

    getBoneColor(parentIdx: number): number {
        if (parentIdx <= 3 || (parentIdx >= 14 && parentIdx <= 16)) {
            return 0xff0000; // Right side - red
        } else if ((parentIdx >= 4 && parentIdx <= 6) || (parentIdx >= 11 && parentIdx <= 13)) {
            return 0x0000ff; // Left side - blue
        }
        return 0x888888; // Center - gray
    }
};