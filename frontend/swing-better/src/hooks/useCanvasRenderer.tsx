// hooks/useCanvasRenderer.ts
import { useEffect } from 'react';
import { ApiResponse } from '@/types';
import { COCO_CONNECTIONS } from '@/lib/constants';

export const useCanvasRenderer = (
    results: ApiResponse | null,
    showJoints: boolean,
    canvasRefs: React.RefObject<Record<string, HTMLCanvasElement | null>>
) => {
    useEffect(() => {
        if (!results?.event_frames || !results?.joints) return;

        Object.entries(results.event_frames).forEach(([idx, frameData]) => {
            const canvas = canvasRefs.current[idx];
            if (!canvas) return;

            const ctx = canvas.getContext('2d');
            if (!ctx) return;

            const img = new Image();
            img.onload = () => {
                canvas.width = img.width;
                canvas.height = img.height;
                ctx.drawImage(img, 0, 0);

                if (showJoints && results.joints?.[idx]) {
                    const joints = results.joints[idx].joints;

                    // Draw connections
                    ctx.strokeStyle = '#00FF00';
                    ctx.lineWidth = 3;
                    COCO_CONNECTIONS.forEach(([start, end]) => {
                        const [x1, y1] = joints[start];
                        const [x2, y2] = joints[end];
                        if (x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0) {
                            ctx.beginPath();
                            ctx.moveTo(x1, y1);
                            ctx.lineTo(x2, y2);
                            ctx.stroke();
                        }
                    });

                    // Draw joints
                    ctx.fillStyle = '#FF0000';
                    joints.forEach(([x, y]) => {
                        if (x > 0 && y > 0) {
                            ctx.beginPath();
                            ctx.arc(x, y, 5, 0, 2 * Math.PI);
                            ctx.fill();
                        }
                    });
                }
            };
            img.src = `data:image/jpeg;base64,${frameData.image}`;
        });
    }, [results, showJoints, canvasRefs]);
};