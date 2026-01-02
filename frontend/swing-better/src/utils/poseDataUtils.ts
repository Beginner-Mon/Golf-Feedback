import * as THREE from 'three';
import { PoseFrameData } from '@/types';
import { SKELETON_CONFIG } from '@/lib/skeleton';

export const processPoseFrame = (frameData: PoseFrameData, scale: number = 1): Float32Array => {
    const joints = frameData.joints_3d;
    const flatArray = new Float32Array(joints.length * 3);

    joints.forEach((joint: number[], idx: number) => {
        flatArray[idx * 3] = joint[0] * scale;
        flatArray[idx * 3 + 1] = -joint[1] * scale; // Flip Y
        flatArray[idx * 3 + 2] = -joint[2] * scale; // Flip Z
    });

    return flatArray;
};

export const extractJointPositions = (pose: Float32Array, numJoints: number): number[][] => {
    const positions: number[][] = [];

    for (let i = 0; i < numJoints; i++) {
        positions[i] = [
            pose[i * 3],
            pose[i * 3 + 1],
            pose[i * 3 + 2]
        ];
    }

    return positions;
};