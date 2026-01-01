"use client";
import React, { useState, useRef, useEffect, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Text } from '@react-three/drei';
import * as THREE from 'three';

// Updated skeleton connections based on the provided image connections:
// Looking at the "Person" column connections:
const parents = [
    -1,  // 0: p1 hip (root)
    0,   // 1: p2 r_hip -> connected to p1 hip
    1,   // 2: p3 r_knee -> connected to p2 r_hip
    2,   // 3: p4 r_ankle -> connected to p3 r_knee
    0,   // 4: p5 l_hip -> connected to p1 hip
    4,   // 5: p6 l_knee -> connected to p5 l_hip
    5,   // 6: p7 l_ankle -> connected to p6 l_knee
    0,   // 7: p8 spine -> connected to p1 hip
    7,   // 8: p9 neck -> connected to p8 spine
    8,   // 9: p10 neck_base -> connected to p9 neck
    9,   // 10: p11 head -> connected to p10 neck_base
    8,   // 11: p12 l_shoulder -> connected to p9 neck (via neck_base connection)
    11,  // 12: p13 l_elbow -> connected to p12 l_shoulder
    12,  // 13: p14 l_wrist -> connected to p13 l_elbow
    8,   // 14: p15 r_shoulder -> connected to p9 neck (via neck_base connection)
    14,  // 15: p16 r_elbow -> connected to p15 r_shoulder
    15   // 16: p17 r_wrist -> connected to p16 r_elbow
];

// Bone connections array - each entry is [parentIndex, childIndex]
const boneConnections = [
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
];

// Colors: right side (red), left side (blue), center (gray)
const jointColors = [
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
];

// Joint labels based on your image table
const jointLabels = [
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
];

interface Pose3DProps {
    poses: Float32Array[];
    fps?: number;
    autoPlay?: boolean;
    showTrajectory?: boolean;
    scale?: number;
    currentFrame: number;
    numJoints: number;
}

const Pose3DSkeleton: React.FC<Pose3DProps> = ({
    poses,
    fps = 30,
    autoPlay = true,
    showTrajectory = true,
    scale = 1,
    currentFrame,
    numJoints
}) => {
    const groupRef = useRef<THREE.Group>(null);
    const [jointPositions, setJointPositions] = useState<number[][]>([]);
    const [boneLines, setBoneLines] = useState<JSX.Element[]>([]);

    const pointGeometry = useMemo(() => {
        const geom = new THREE.BufferGeometry();
        const positions = new Float32Array(numJoints * 3);
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        const colors = new Float32Array(numJoints * 3);
        geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        return geom;
    }, [numJoints]);

    const trajectoryGeometry = useMemo(() => {
        const geom = new THREE.BufferGeometry();
        const positions = new Float32Array(poses.length * 3);
        geom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        return geom;
    }, [poses.length]);

    // Update joint positions and bones when currentFrame changes
    useEffect(() => {
        if (!poses.length || currentFrame >= poses.length) return;

        const currentPose = poses[currentFrame];
        const scaledPose = new Float32Array(numJoints * 3);
        const newJointPositions: number[][] = [];

        for (let i = 0; i < numJoints; i++) {
            const x = currentPose[i * 3];
            const y = currentPose[i * 3 + 1];
            const z = currentPose[i * 3 + 2];

            // (x, y, -z)
            scaledPose[i * 3] = x * scale;
            scaledPose[i * 3 + 1] = -y * scale;     // giữ Y là height
            scaledPose[i * 3 + 2] = -z * scale;    // lật Z

            newJointPositions[i] = [x * scale, -y * scale, -z * scale];
        }

        setJointPositions(newJointPositions);

        pointGeometry.attributes.position.array.set(scaledPose);
        pointGeometry.attributes.position.needsUpdate = true;

        for (let i = 0; i < numJoints; i++) {
            const color = new THREE.Color(jointColors[i] || 0x888888);
            pointGeometry.attributes.color.setXYZ(i, color.r, color.g, color.b);
        }
        pointGeometry.attributes.color.needsUpdate = true;

        if (showTrajectory) {
            const trajPositions = new Float32Array(poses.length * 3);
            for (let f = 0; f < poses.length; f++) {
                const p = poses[f];
                trajPositions[f * 3] = p[0] * scale;
                trajPositions[f * 3 + 1] = p[1] * scale;
                trajPositions[f * 3 + 2] = p[2] * scale;
            }
            const posAttr = trajectoryGeometry.attributes.position as THREE.BufferAttribute;
            posAttr.array.set(trajPositions);
            posAttr.needsUpdate = true;
        }

        // Create bone lines
        const newBoneLines = boneConnections.map(([parentIdx, childIdx], index) => {
            if (parentIdx >= newJointPositions.length || childIdx >= newJointPositions.length) {
                return null;
            }

            const parentPos = newJointPositions[parentIdx];
            const childPos = newJointPositions[childIdx];

            if (!parentPos || !childPos) {
                return null;
            }

            const lineGeometry = new THREE.BufferGeometry().setFromPoints([
                new THREE.Vector3(parentPos[0], parentPos[1], parentPos[2]),
                new THREE.Vector3(childPos[0], childPos[1], childPos[2])
            ]);

            // Determine bone color based on which side it's on
            let boneColor = 0x888888; // Default gray for center
            if (parentIdx <= 3 || (parentIdx >= 14 && parentIdx <= 16)) {
                boneColor = 0xff0000; // Right side - red
            } else if ((parentIdx >= 4 && parentIdx <= 6) || (parentIdx >= 11 && parentIdx <= 13)) {
                boneColor = 0x0000ff; // Left side - blue
            }

            return (
                <line key={`bone-${index}`} geometry={lineGeometry}>
                    <lineBasicMaterial color={boneColor} linewidth={2} />
                </line>
            );
        }).filter(Boolean) as JSX.Element[];

        setBoneLines(newBoneLines);

    }, [currentFrame, poses, scale, showTrajectory, pointGeometry, trajectoryGeometry, numJoints]);

    const { camera } = useThree();
    useEffect(() => {
        camera.position.set(2, 1, 3);
        camera.lookAt(0, 1, 0);
    }, [camera]);

    return (
        <group ref={groupRef}>
            {/* Render bones as lines */}
            {boneLines}

            {/* Render joints as points */}
            <points geometry={pointGeometry}>
                <pointsMaterial size={0.15} vertexColors sizeAttenuation />
            </points>

            {/* Render trajectory if enabled */}
            {showTrajectory && (
                <primitive object={new THREE.Line(trajectoryGeometry, new THREE.LineBasicMaterial({ color: 0x00ff00, opacity: 0.5, transparent: true }))} />
            )}

            {/* Joint labels with names */}
            {jointPositions.length > 0 && jointPositions.map((pos, i) => (
                <Text
                    key={`label-${i}`}
                    position={[pos[0], pos[1] + 0.08, pos[2]]}
                    fontSize={0.05}
                    color="white"
                    anchorX="center"
                    anchorY="middle"
                    outlineWidth={0.01}
                    outlineColor="black"
                >
                    {i}: {jointLabels[i]}
                </Text>
            ))}
        </group>
    );
};

const Visualisation3D: React.FC = () => {
    const [poses, setPoses] = useState<Float32Array[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [currentFrame, setCurrentFrame] = useState(0);
    const [isPlaying, setIsPlaying] = useState(true);
    const [totalFrames, setTotalFrames] = useState(0);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [numJoints, setNumJoints] = useState(17);
    const [metadata, setMetadata] = useState<any>(null);
    const animationRef = useRef<number | null>(null);

    const fetchPoseData = async (page: number = 1) => {
        try {
            setLoading(true);
            setError(null);

            const response = await fetch(`http://127.0.0.1:8000/analyze/3d?page=${page}&page_size=30`);

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();

            if (result.status === 'success' && result.data) {
                if (result.meta) {
                    setMetadata(result.meta);
                    setNumJoints(result.meta.joints);
                }

                const newPoses: Float32Array[] = result.data.map((frameData: any) => {
                    const joints = frameData.joints_3d;
                    const flatArray = new Float32Array(joints.length * 3);
                    joints.forEach((joint: number[], idx: number) => {
                        flatArray[idx * 3] = joint[0];
                        flatArray[idx * 3 + 1] = joint[1];
                        flatArray[idx * 3 + 2] = joint[2];
                    });
                    return flatArray;
                });

                setPoses(prev => page === 1 ? newPoses : [...prev, ...newPoses]);
                setTotalFrames(result.pagination.total_frames);
                setTotalPages(result.pagination.total_pages);
                setCurrentPage(result.pagination.page);
            } else {
                throw new Error('Invalid response format');
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch pose data');
            console.error('Error fetching pose data:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchPoseData(1);
    }, []);

    useEffect(() => {
        if (isPlaying && poses.length > 0) {
            const fps = 30;
            const interval = 1000 / fps;

            animationRef.current = window.setInterval(() => {
                setCurrentFrame(prev => {
                    const nextFrame = (prev + 1) % poses.length;

                    if (nextFrame === 0 && currentPage < totalPages) {
                        fetchPoseData(currentPage + 1);
                    }

                    return nextFrame;
                });
            }, interval);

            return () => {
                if (animationRef.current) clearInterval(animationRef.current);
            };
        }
    }, [isPlaying, poses.length, currentPage, totalPages]);

    const togglePlayPause = () => setIsPlaying(!isPlaying);

    const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const frame = parseInt(e.target.value);
        setCurrentFrame(frame);
    };

    if (loading && poses.length === 0) {
        return (
            <div className="w-full h-screen flex items-center justify-center bg-black text-white">
                <div className="text-center">
                    <div className="animate-spin rounded-full h-16 w-16 border-t-2 border-b-2 border-white mx-auto mb-4"></div>
                    <p className="text-xl">Loading pose data...</p>
                </div>
            </div>
        );
    }

    if (error) {
        return (
            <div className="w-full h-screen flex items-center justify-center bg-black text-white">
                <div className="text-center max-w-md">
                    <h2 className="text-2xl font-bold mb-4 text-red-500">Error</h2>
                    <p className="mb-4">{error}</p>
                    <button
                        onClick={() => fetchPoseData(1)}
                        className="px-6 py-2 bg-blue-600 hover:bg-blue-700 rounded"
                    >
                        Retry
                    </button>
                </div>
            </div>
        );
    }

    return (
        <div className="flex-1 overflow-hidden w-full h-screen flex flex-col bg-black">
            <Canvas>
                <PerspectiveCamera makeDefault position={[2, 1, 4]} />
                <ambientLight intensity={0.8} />
                <directionalLight position={[5, 10, 5]} intensity={1} castShadow />
                <OrbitControls enablePan enableZoom enableRotate />
                <primitive object={new THREE.AxesHelper(1.5)} />

                <Pose3DSkeleton
                    poses={poses}
                    fps={30}
                    autoPlay={false}
                    scale={1}
                    currentFrame={currentFrame}
                    numJoints={numJoints}
                />

                <mesh rotation={[-Math.PI / 2, 0, Math.PI / 2]} position={[0, 3, 0]} receiveShadow>
                    <planeGeometry args={[10, 10]} />
                    <meshStandardMaterial color="#444444" />
                </mesh>
            </Canvas>

            <div className="p-4 text-white bg-gray-900">
                <h2 className="text-xl font-bold text-center mb-1">3D Pose Visualization</h2>
                {metadata && (
                    <p className="text-sm opacity-70 text-center mb-2">
                        Custom Skeleton Format • {metadata.total_frames} frames • {metadata.joints} joints
                    </p>
                )}
                <p className="text-xs opacity-60 text-center mb-4">Drag to rotate • Scroll to zoom</p>

                <div className="flex items-center gap-4 mb-2">
                    <button
                        onClick={togglePlayPause}
                        className="px-4 py-2 bg-blue-600 hover:bg-blue-700 rounded font-medium"
                    >
                        {isPlaying ? 'Pause' : 'Play'}
                    </button>

                    <div className="flex-1">
                        <input
                            type="range"
                            min="0"
                            max={Math.max(0, poses.length - 1)}
                            value={currentFrame}
                            onChange={handleSliderChange}
                            className="w-full"
                        />
                    </div>

                    <span className="text-sm whitespace-nowrap">
                        {currentFrame + 1} / {totalFrames}
                    </span>
                </div>

                {loading && currentPage > 1 && (
                    <p className="text-xs text-center opacity-70">Loading more frames...</p>
                )}
            </div>
        </div>
    );
};

export default Visualisation3D;