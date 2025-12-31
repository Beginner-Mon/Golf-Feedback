"use client";
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';

// Common human skeleton parent array (COCO-style 17 joints, root at pelvis=0)
const parents = [-1, 0, 1, 2, 0, 4, 5, 0, 7, 8, 9, 8, 11, 12, 8, 13, 14];

// Colors: left (blue), right (red), center (gray)
const jointColors = [
    0x888888, // 0 pelvis
    0x888888, // 1 thorax
    0x888888, // 2 neck
    0x888888, // 3 head
    0x0000ff, // 4 left_shoulder
    0x0000ff, // 5 left_elbow
    0x0000ff, // 6 left_wrist
    0xff0000, // 7 right_shoulder
    0xff0000, // 8 right_elbow
    0xff0000, // 9 right_wrist
    0x0000ff, // 10 left_hip
    0x0000ff, // 11 left_knee
    0x0000ff, // 12 left_ankle
    0xff0000, // 13 right_hip
    0xff0000, // 14 right_knee
    0xff0000, // 15 right_ankle
];

// Props for the component
interface Pose3DProps {
    poses: Float32Array[]; // Array of frames: each frame is Float32Array(num_joints * 3)
    fps?: number;          // Playback FPS (default 30)
    autoPlay?: boolean;    // Auto play animation
    showTrajectory?: boolean;
    scale?: number;        // Scale factor for the pose
}

const Pose3DSkeleton: React.FC<Pose3DProps> = ({
    poses,
    fps = 30,
    autoPlay = true,
    showTrajectory = true,
    scale = 1,
}) => {
    const groupRef = useRef<THREE.Group>(null);
    const [frame, setFrame] = useState(0);

    // Precompute geometry for lines and points
    const lineGeometry = useMemo(() => {
        const geom = new THREE.BufferGeometry();
        const positions = new Float32Array(parents.length * 2 * 3);
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        return geom;
    }, []);

    const pointGeometry = useMemo(() => {
        const geom = new THREE.BufferGeometry();
        const positions = new Float32Array(parents.length * 3);
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const colors = new Float32Array(parents.length * 3);
        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        return geom;
    }, []);

    // Trajectory geometry
    const trajectoryGeometry = useMemo(() => {
        const geom = new THREE.BufferGeometry();
        const positions = new Float32Array(poses.length * 3);
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        return geom;
    }, [poses.length]);

    // Update skeleton for current frame
    useEffect(() => {
        if (!poses.length) return;

        const currentPose = poses[frame];
        const scaledPose = currentPose.map(v => v * scale);

        // Update points
        pointGeometry.attributes.position.array.set(scaledPose);
        pointGeometry.attributes.position.needsUpdate = true;

        // Update colors (vertex colors)
        for (let i = 0; i < parents.length; i++) {
            const color = new THREE.Color(jointColors[i] || 0x888888);
            pointGeometry.attributes.color.setXYZ(i, color.r, color.g, color.b);
        }
        pointGeometry.attributes.color.needsUpdate = true;

        // Update lines
        const linePos = lineGeometry.attributes.position.array as Float32Array;
        let idx = 0;
        for (let j = 0; j < parents.length; j++) {
            const p = parents[j];
            if (p === -1) continue;
            // parent joint
            linePos[idx++] = scaledPose[p * 3];
            linePos[idx++] = scaledPose[p * 3 + 1];
            linePos[idx++] = scaledPose[p * 3 + 2];
            // child joint
            linePos[idx++] = scaledPose[j * 3];
            linePos[idx++] = scaledPose[j * 3 + 1];
            linePos[idx++] = scaledPose[j * 3 + 2];
        }
        lineGeometry.attributes.position.needsUpdate = true;

        // Trajectory (pelvis/hip movement)
        if (showTrajectory) {
            const trajPositions = new Float32Array(poses.length * 3);
            for (let f = 0; f < poses.length; f++) {
                const p = poses[f];
                trajPositions[f * 3] = p[0] * scale;     // assuming pelvis is joint 0: x
                trajPositions[f * 3 + 1] = p[1] * scale; // y
                trajPositions[f * 3 + 2] = p[2] * scale; // z
            }
            const posAttr = trajectoryGeometry.attributes.position as THREE.BufferAttribute;
            posAttr.array.set(trajPositions);
            posAttr.needsUpdate = true;
        }
    }, [frame, poses, scale, showTrajectory, lineGeometry, pointGeometry, trajectoryGeometry]);

    // Animation loop
    useFrame((state, delta) => {
        if (autoPlay && poses.length > 1) {
            setFrame(prev => (prev + 1) % poses.length);
        }
    });

    // Camera auto-focus on the pose
    const { camera } = useThree();
    useEffect(() => {
        camera.position.set(0, 1, 3);
        camera.lookAt(0, 1, 0);
    }, [camera]);

    return (
        <group ref={groupRef}>
            {/* Skeleton lines */}
            <lineSegments geometry={lineGeometry}>
                <lineBasicMaterial color={0xffffff} linewidth={5} />
            </lineSegments>

            {/* Joint points */}
            <points geometry={pointGeometry}>
                <pointsMaterial size={0.05} vertexColors sizeAttenuation />
            </points>

            {/* Optional trajectory */}
            {showTrajectory && (
                <primitive object={new THREE.Line(trajectoryGeometry, new THREE.LineBasicMaterial({ color: 0x00ff00, opacity: 0.5, transparent: true }))} />
            )}
        </group>
    );
};

const Visualisation3D: React.FC = () => {
    // Example dummy data - replace with your actual 3D poses
    // poses: array of frames, each frame Float32Array(17 joints * 3 coords)
    const examplePoses: Float32Array[] = useMemo(() => {
        const numFrames = 100;
        const numJoints = 17;
        const poses: Float32Array[] = [];
        for (let f = 0; f < numFrames; f++) {
            const pose = new Float32Array(numJoints * 3);
            const t = f / numFrames * Math.PI * 2;
            // Simple walking-like animation (arms/legs swing)
            pose[0 * 3] = 0; // pelvis x
            pose[0 * 3 + 1] = 1; // pelvis y
            pose[0 * 3 + 2] = 0; // pelvis z

            // Add some basic joint positions for visibility
            pose[1 * 3 + 1] = 1.2; // thorax
            pose[2 * 3 + 1] = 1.4; // neck
            pose[3 * 3 + 1] = 1.6; // head

            // Arms swing
            pose[4 * 3] = -0.3; // left shoulder x
            pose[4 * 3 + 1] = 1.2 + Math.sin(t) * 0.1; // left shoulder y
            pose[5 * 3] = -0.4; // left elbow x
            pose[5 * 3 + 1] = 1.0 + Math.sin(t) * 0.2; // left elbow y

            pose[7 * 3] = 0.3; // right shoulder x
            pose[7 * 3 + 1] = 1.2 + Math.sin(t + Math.PI) * 0.1; // right shoulder y
            pose[8 * 3] = 0.4; // right elbow x
            pose[8 * 3 + 1] = 1.0 + Math.sin(t + Math.PI) * 0.2; // right elbow y

            // Legs
            pose[10 * 3] = -0.15; // left hip x
            pose[10 * 3 + 1] = 1.0; // left hip y
            pose[11 * 3] = -0.15; // left knee x
            pose[11 * 3 + 1] = 0.5; // left knee y
            pose[12 * 3] = -0.15; // left ankle x
            pose[12 * 3 + 1] = 0.0; // left ankle y

            pose[13 * 3] = 0.15; // right hip x
            pose[13 * 3 + 1] = 1.0; // right hip y
            pose[14 * 3] = 0.15; // right knee x
            pose[14 * 3 + 1] = 0.5; // right knee y
            pose[15 * 3] = 0.15; // right ankle x
            pose[15 * 3 + 1] = 0.0; // right ankle y

            poses.push(pose);
        }
        return poses;
    }, []);

    return (
        <div className="flex-1 overflow-hidden w-full h-screen flex flex-col bg-black">
            <Canvas>
                <PerspectiveCamera makeDefault position={[0, 1, 4]} />
                <ambientLight intensity={0.8} />
                <directionalLight position={[5, 10, 5]} intensity={1} castShadow />
                <OrbitControls enablePan enableZoom enableRotate />

                <Pose3DSkeleton poses={examplePoses} fps={30} autoPlay scale={1} />

                {/* Ground plane for reference */}
                <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.1, 0]} receiveShadow>
                    <planeGeometry args={[10, 10]} />
                    <meshStandardMaterial color="#444444" />
                </mesh>
            </Canvas>

            <div className="p-4 text-white text-center bg-gray-900">
                <h2 className="text-xl font-bold">3D Pose Visualisation</h2>
                <p className="text-sm opacity-70">Drag to rotate • Scroll to zoom</p>
            </div>
        </div>
    );
};

export default Visualisation3D;