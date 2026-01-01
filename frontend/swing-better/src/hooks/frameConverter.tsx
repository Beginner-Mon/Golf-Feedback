import type { Pose3DFrame } from '../types/index';

export function convertFramesToPoses(frames: Pose3DFrame[]): Float32Array[] {
    return frames.map(f => {
        const flat = new Float32Array(f.joints_3d.length * 3);
        f.joints_3d.forEach((j, i) => {
            flat[i * 3] = j[0];
            flat[i * 3 + 1] = j[1];
            flat[i * 3 + 2] = j[2];
        });
        return flat;
    });
}