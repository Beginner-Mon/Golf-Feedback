import React from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { Pose3DSkeleton } from './Pose3DSkeleton';
import { usePoseAnimation } from '@/hooks/usePoseAnimation';

const Visualization3D: React.FC = () => {
    const {
        poses,
        loading,
        error,
        currentFrame,
        isPlaying,
        totalFrames,
        numJoints,
        metadata,
        togglePlayPause,
        setCurrentFrame,
        fetchMoreData
    } = usePoseAnimation(1);

    const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const frame = parseInt(e.target.value);
        setCurrentFrame(frame);
    };

    const handleRetry = () => {
        fetchMoreData(1);
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
                        onClick={handleRetry}
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
            </div>
        </div>
    );
};

export default Visualization3D;