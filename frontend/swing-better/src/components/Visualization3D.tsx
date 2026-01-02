import React, { useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera } from '@react-three/drei';
import * as THREE from 'three';
import { Pose3DSkeleton } from './Pose3DSkeleton';
import { usePoseAnimation } from '@/hooks/usePoseAnimation';

const Visualisation3D: React.FC = () => {
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

    // State for toggling joint labels
    const [showJointLabels, setShowJointLabels] = useState(true);

    const handleSliderChange = (e: React.ChangeEvent<HTMLInputElement>) => {
        const frame = parseInt(e.target.value);
        setCurrentFrame(frame);
    };

    const handleRetry = () => {
        fetchMoreData(1);
    };

    const toggleJointLabels = () => {
        setShowJointLabels(!showJointLabels);
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
                    showJointLabels={showJointLabels}  // Pass the prop
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

                {/* Control buttons */}
                <div className="flex flex-wrap gap-3 mb-4">
                    <button
                        onClick={togglePlayPause}
                        className={`px-4 py-2 rounded font-medium flex items-center gap-2 ${isPlaying ? 'bg-yellow-600 hover:bg-yellow-700' : 'bg-green-600 hover:bg-green-700'
                            }`}
                    >
                        {isPlaying ? (
                            <>
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zM7 8a1 1 0 012 0v4a1 1 0 11-2 0V8zm5-1a1 1 0 00-1 1v4a1 1 0 102 0V8a1 1 0 00-1-1z" clipRule="evenodd" />
                                </svg>
                                Pause
                            </>
                        ) : (
                            <>
                                <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                                    <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zM9.555 7.168A1 1 0 008 8v4a1 1 0 001.555.832l3-2a1 1 0 000-1.664l-3-2z" clipRule="evenodd" />
                                </svg>
                                Play
                            </>
                        )}
                    </button>

                    <button
                        onClick={toggleJointLabels}
                        className={`px-4 py-2 rounded font-medium flex items-center gap-2 ${showJointLabels ? 'bg-blue-600 hover:bg-blue-700' : 'bg-gray-700 hover:bg-gray-600'
                            }`}
                    >
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M12.586 4.586a2 2 0 112.828 2.828l-3 3a2 2 0 01-2.828 0 1 1 0 00-1.414 1.414 4 4 0 005.656 0l3-3a4 4 0 00-5.656-5.656l-1.5 1.5a1 1 0 101.414 1.414l1.5-1.5zm-5 5a2 2 0 012.828 0 1 1 0 101.414-1.414 4 4 0 00-5.656 0l-3 3a4 4 0 105.656 5.656l1.5-1.5a1 1 0 10-1.414-1.414l-1.5 1.5a2 2 0 11-2.828-2.828l3-3z" clipRule="evenodd" />
                        </svg>
                        {showJointLabels ? 'Hide Labels' : 'Show Labels'}
                    </button>

                    <button
                        onClick={() => setCurrentFrame(0)}
                        className="px-4 py-2 bg-purple-600 hover:bg-purple-700 rounded font-medium flex items-center gap-2"
                    >
                        <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 20 20">
                            <path fillRule="evenodd" d="M10 18a8 8 0 100-16 8 8 0 000 16zm.707-10.293a1 1 0 00-1.414-1.414l-3 3a1 1 0 000 1.414l3 3a1 1 0 001.414-1.414L9.414 11H13a1 1 0 100-2H9.414l1.293-1.293z" clipRule="evenodd" />
                        </svg>
                        Reset Frame
                    </button>
                </div>

                {/* Frame slider */}
                <div className="flex items-center gap-4 mb-2">
                    <span className="text-sm whitespace-nowrap min-w-[60px]">Frame:</span>

                    <div className="flex-1">
                        <input
                            type="range"
                            min="0"
                            max={Math.max(0, poses.length - 1)}
                            value={currentFrame}
                            onChange={handleSliderChange}
                            className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                        />
                    </div>

                    <span className="text-sm whitespace-nowrap min-w-[100px]">
                        {currentFrame + 1} / {totalFrames}
                    </span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-gray-700 rounded-full h-1.5 mb-2">
                    <div
                        className="bg-blue-600 h-1.5 rounded-full"
                        style={{ width: `${((currentFrame + 1) / totalFrames) * 100}%` }}
                    />
                </div>

                <div className="flex justify-between text-xs text-gray-400">
                    <span>Start</span>
                    <span>Progress: {Math.round(((currentFrame + 1) / totalFrames) * 100)}%</span>
                    <span>End</span>
                </div>
            </div>
        </div>
    );
};

export default Visualisation3D;