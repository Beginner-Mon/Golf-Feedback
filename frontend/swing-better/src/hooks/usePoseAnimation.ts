import { useState, useEffect, useRef, useCallback } from 'react';
import { PoseApiResponse, PaginationData, MetaData } from '@/types';
import { fetchPoseData } from '@/services/api';
import { processPoseFrame } from '@/utils/poseDataUtils';

// Simple debounce function
function debounce<T extends (...args: any[]) => any>(func: T, wait: number): T {
    let timeout: NodeJS.Timeout | null = null;

    return ((...args: Parameters<T>) => {
        if (timeout) clearTimeout(timeout);
        timeout = setTimeout(() => {
            func(...args);
        }, wait);
    }) as T;
}

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
    const isFetchingRef = useRef<boolean>(false);
    const lastFetchedPageRef = useRef<Set<number>>(new Set());

    const togglePlayPause = () => setIsPlaying(!isPlaying);

    // Debounced fetch function
    const fetchMoreData = useCallback(
        debounce(async (page: number) => {
            // Skip if already fetched this page
            if (lastFetchedPageRef.current.has(page)) {
                console.log(`Page ${page} already fetched, skipping...`);
                return;
            }

            // Skip if already fetching
            if (isFetchingRef.current) {
                console.log('Already fetching, skipping...');
                return;
            }

            try {
                isFetchingRef.current = true;
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

                    // Mark this page as fetched
                    lastFetchedPageRef.current.add(page);
                }
            } catch (err) {
                setError(err instanceof Error ? err.message : 'Failed to fetch pose data');
                console.error('Error fetching pose data:', err);
            } finally {
                setLoading(false);
                isFetchingRef.current = false;
            }
        }, 100), // 100ms debounce
        []
    );

    // Initial fetch
    useEffect(() => {
        fetchMoreData(initialPage);

        return () => {
            if (animationRef.current) {
                clearInterval(animationRef.current);
            }
        };
    }, [initialPage, fetchMoreData]);

    // Animation effect
    useEffect(() => {
        if (!isPlaying || poses.length === 0) {
            if (animationRef.current) {
                clearInterval(animationRef.current);
                animationRef.current = null;
            }
            return;
        }

        const fps = 30;
        const interval = 1000 / fps;

        animationRef.current = window.setInterval(() => {
            setCurrentFrame(prev => {
                const nextFrame = prev + 1;

                // If we've reached the end of current poses
                if (nextFrame >= poses.length) {
                    // If there are more pages to fetch
                    if (currentPage < totalPages) {
                        const nextPage = currentPage + 1;
                        fetchMoreData(nextPage);
                    }
                    // Loop back to start
                    return 0;
                }

                return nextFrame;
            });
        }, interval);

        return () => {
            if (animationRef.current) {
                clearInterval(animationRef.current);
                animationRef.current = null;
            }
        };
    }, [isPlaying, poses.length, currentPage, totalPages, fetchMoreData]);

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