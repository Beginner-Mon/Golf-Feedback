// components/EventFrames.tsx
import React from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { ApiResponse } from '@/types';

interface EventFramesProps {
    results: ApiResponse;
    showJoints: boolean;
    selectedFrameIdx: string | null;
    canvasRefs: React.RefObject<Record<string, HTMLCanvasElement | null>>;
    onToggleJoints: () => void;
    onSelectFrame: (idx: string) => void;
}

export const EventFrames: React.FC<EventFramesProps> = ({
    results,
    showJoints,
    selectedFrameIdx,
    canvasRefs,
    onToggleJoints,
    onSelectFrame,
}) => {
    if (!results.event_frames) return null;

    return (
        <div className="h-screen flex flex-col bg-white border-t border-gray-200">
            <div className="flex items-center justify-between p-3 border-b border-gray-200">
                <h2 className="text-base font-bold text-gray-800">Swing Events</h2>
                <button
                    onClick={onToggleJoints}
                    className="flex items-center space-x-2 px-3 py-1 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm transition"
                >
                    {showJoints ? <Eye className="w-4 h-4" /> : <EyeOff className="w-4 h-4" />}
                    <span>{showJoints ? 'Hide Joints' : 'Show Joints'}</span>
                </button>
            </div>

            {/* Frames Area */}
            <div className="flex-1 p-10  flex items-center justify-center">
                <div className="flex items-center overflow-x-auto gap-3">
                    {Object.entries(results.event_frames).map(([idx, frameData]) => (
                        <div
                            key={idx}
                            className={`w-fit transition`}
                        >
                            {/* FRAME */}
                            <div className="w-[24rem] h-[32rem] overflow-hidden flex items-center justify-center">
                                <canvas
                                    ref={(el) => {
                                        canvasRefs.current[idx] = el;
                                    }}
                                    className=" object-contain shadow-lg bg-black"
                                />
                            </div>

                            {/* LABEL */}
                            <p className="text-sm text-center mt-3 font-semibold text-gray-700">
                                {frameData.event}
                            </p>
                        </div>
                    ))}
                </div>
            </div>

        </div>
    );
};