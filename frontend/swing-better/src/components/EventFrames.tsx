// components/EventFrames.tsx
import React from 'react';
import { Eye, EyeOff } from 'lucide-react';
import { ApiResponse } from '@/types';

interface EventFramesProps {
    results: ApiResponse;
    showJoints: boolean;
    selectedFrameIdx: string | null;
    canvasRefs: React.MutableRefObject<Record<string, HTMLCanvasElement | null>>;
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
        <div className="flex-shrink-0 bg-white border-t border-gray-200" style={{ height: '35vh' }}>
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

            <div className="overflow-x-auto overflow-y-hidden p-3" style={{ height: 'calc(35vh - 60px)' }}>
                <div className="flex space-x-3">
                    {Object.entries(results.event_frames).map(([idx, frameData]) => (
                        <div
                            key={idx}
                            className={`flex-shrink-0 cursor-pointer transition ${selectedFrameIdx === idx ? 'ring-4 ring-blue-500' : ''
                                }`}
                            onClick={() => onSelectFrame(idx)}
                        >
                            <canvas
                                ref={(el) => { canvasRefs.current[idx] = el; }}
                                className="rounded-lg shadow-md"
                                style={{ height: 'calc(35vh - 100px)', width: 'auto' }}
                            />
                            <p className="text-xs text-center mt-1 font-semibold text-gray-700">
                                {frameData.event}
                            </p>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};