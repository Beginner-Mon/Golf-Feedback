import { useState, useEffect, useRef } from 'react';
import { fetchPoseData, } from '@/services/api';
import { PoseApiResponse, MetaData, PaginationData } from '@/types';
import { processPoseFrame } from '@/utils/poseDataUtils';

interface UsePoseAnimationReturn {
    poses: Float32Array[];
    loading: boolean;
    error: string | null;
    currentFrame: number;
    isPlaying: boolean;
    totalFrames: number;
    currentPage: number;
    totalPages: number;
    numJoints: number;
    metadata: MetaData | null;
    togglePlayPause: () => void;
    setCurrentFrame: (frame: number) => void;
    fetchMoreData: (page: number) => Promise<void>;
}

export const usePoseAnimation = (initialPage: number = 1): UsePoseAnimationReturn => {
    const [poses, setPoses] = useState<Float32Array[]>([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [currentFrame, setCurrentFrame] = useState(0);
    const [isPlaying, setIsPlaying] = useState(true);
    const [totalFrames, setTotalFrames] = useState(0);
    const [currentPage, setCurrentPage] = useState(1);
    const [totalPages, setTotalPages] = useState(1);
    const [numJoints, setNumJoints] = useState(17);
    const [metadata, setMetadata] = useState<MetaData | null>(null);
    const animationRef = useRef<number | null>(null);

    const togglePlayPause = () => setIsPlaying(!isPlaying);

    const fetchMoreData = async (page: number) => {
        try {
            setLoading(true);
            const result = await fetchPoseData(page);

            if (result.status === 'success' && result.data) {
                if (result.meta) {
                    setMetadata(result.meta);
                    setNumJoints(result.meta.joints);
                }

                const newPoses = result.data.map(frameData => processPoseFrame(frameData));

                setPoses(prev => page === 1 ? newPoses : [...prev, ...newPoses]);
                setTotalFrames(result.pagination.total_frames);
                setTotalPages(result.pagination.total_pages);
                setCurrentPage(result.pagination.page);
            }
        } catch (err) {
            setError(err instanceof Error ? err.message : 'Failed to fetch pose data');
            console.error('Error fetching pose data:', err);
        } finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        fetchMoreData(initialPage);
    }, []);

    useEffect(() => {
        if (isPlaying && poses.length > 0) {
            const fps = 30;
            const interval = 1000 / fps;

            animationRef.current = window.setInterval(() => {
                setCurrentFrame(prev => {
                    const nextFrame = (prev + 1) % poses.length;

                    if (nextFrame === 0 && currentPage < totalPages) {
                        fetchMoreData(currentPage + 1);
                    }

                    return nextFrame;
                });
            }, interval);

            return () => {
                if (animationRef.current) clearInterval(animationRef.current);
            };
        }
    }, [isPlaying, poses.length, currentPage, totalPages]);

    return {
        poses,
        loading,
        error,
        currentFrame,
        isPlaying,
        totalFrames,
        currentPage,
        totalPages,
        numJoints,
        metadata,
        togglePlayPause,
        setCurrentFrame,
        fetchMoreData
    };
};