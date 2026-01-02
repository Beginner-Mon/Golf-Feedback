import React, { useState, useRef, useEffect, useMemo } from 'react';
import * as THREE from 'three';
import { Text } from '@react-three/drei';
import { SKELETON_CONFIG } from '@/lib/skeleton';
import { extractJointPositions } from '@/utils/poseDataUtils';

interface Pose3DProps {
    poses: Float32Array[];
    fps?: number;
    autoPlay?: boolean;
    showTrajectory?: boolean;
    showJointLabels?: boolean;
    scale?: number;
    currentFrame: number;
    numJoints: number;
}

export const Pose3DSkeleton: React.FC<Pose3DProps> = ({
    poses,
    showTrajectory = true,
    showJointLabels = true,
    scale = 1,
    currentFrame,
    numJoints
}) => {
    const groupRef = useRef<THREE.Group>(null);
    const [jointPositions, setJointPositions] = useState<number[][]>([]);

    // Use state for bone lines
    const [boneLines, setBoneLines] = useState<{ geometry: THREE.BufferGeometry, material: THREE.LineBasicMaterial }[]>([]);

    const pointGeometry = useMemo(() => {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(numJoints * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

        const colors = new Float32Array(numJoints * 3);
        geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));

        return geometry;
    }, [numJoints]);

    const trajectoryGeometry = useMemo(() => {
        const geometry = new THREE.BufferGeometry();
        const positions = new Float32Array(poses.length * 3);
        geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
        return geometry;
    }, [poses.length]);

    const trajectoryLine = useMemo(() => {
        const material = new THREE.LineBasicMaterial({
            color: 0x00ff00,
            opacity: 0.5,
            transparent: true
        });
        return new THREE.Line(trajectoryGeometry, material);
    }, [trajectoryGeometry]);

    const updateJointGeometry = (scaledPose: Float32Array) => {
        pointGeometry.attributes.position.array.set(scaledPose);
        (pointGeometry.attributes.position as THREE.BufferAttribute).needsUpdate = true;

        for (let i = 0; i < numJoints; i++) {
            const color = new THREE.Color(SKELETON_CONFIG.JOINT_COLORS[i] || 0x888888);
            pointGeometry.attributes.color.setXYZ(i, color.r, color.g, color.b);
        }
        (pointGeometry.attributes.color as THREE.BufferAttribute).needsUpdate = true;
    };

    const updateBoneLines = (positions: number[][]) => {
        const newBoneLines: { geometry: THREE.BufferGeometry, material: THREE.LineBasicMaterial }[] = [];

        SKELETON_CONFIG.BONE_CONNECTIONS.forEach(([parentIdx, childIdx]) => {
            if (parentIdx >= positions.length || childIdx >= positions.length) {
                return;
            }

            const parentPos = positions[parentIdx];
            const childPos = positions[childIdx];

            if (!parentPos || !childPos) {
                return;
            }

            // Create geometry for this bone
            const lineGeometry = new THREE.BufferGeometry();
            const vertices = new Float32Array([
                parentPos[0], parentPos[1], parentPos[2],
                childPos[0], childPos[1], childPos[2]
            ]);
            lineGeometry.setAttribute('position', new THREE.BufferAttribute(vertices, 3));

            // Create material with appropriate color
            const boneColor = SKELETON_CONFIG.getBoneColor(parentIdx);
            const material = new THREE.LineBasicMaterial({
                color: boneColor,
                linewidth: 2
            });

            newBoneLines.push({ geometry: lineGeometry, material });
        });

        setBoneLines(newBoneLines);
    };

    useEffect(() => {
        if (!poses.length || currentFrame >= poses.length) return;

        const currentPose = poses[currentFrame];
        const newJointPositions = extractJointPositions(currentPose, numJoints);
        setJointPositions(newJointPositions);

        updateJointGeometry(currentPose);
        updateBoneLines(newJointPositions);

    }, [currentFrame, poses, scale, numJoints]);

    return (
        <group ref={groupRef}>
            {/* Render bones as lines using primitives */}
            {boneLines.map((bone, index) => (
                <primitive
                    key={`bone-${index}`}
                    object={new THREE.Line(bone.geometry, bone.material)}
                />
            ))}

            {/* Render joints as points */}
            <points geometry={pointGeometry}>
                <pointsMaterial size={0.15} vertexColors sizeAttenuation />
            </points>

            {/* Render trajectory if enabled */}
            {showTrajectory && (
                <primitive object={trajectoryLine} />
            )}

            {/* Joint labels with names - conditionally rendered */}
            {showJointLabels && jointPositions.length > 0 && jointPositions.map((pos, i) => (
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
                    {i}: {SKELETON_CONFIG.JOINT_LABELS[i]}
                </Text>
            ))}
        </group>
    );
};