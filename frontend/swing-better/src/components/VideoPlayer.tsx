// components/VideoPlayer.tsx
import React from 'react';
import { Video, X, Activity, AlertCircle } from 'lucide-react';
import { ApiResponse } from '@/types';

interface VideoPlayerProps {
    videoUrl: string;
    file: File;
    loading: boolean;
    results: ApiResponse | null;
    error: string | null;
    onAnalyze: () => void;
    onReset: () => void;
}

export const VideoPlayer: React.FC<VideoPlayerProps> = ({
    videoUrl,
    file,
    loading,
    results,
    error,
    onAnalyze,
    onReset,
}) => {
    return (
        <div className="bg-white border-r border-gray-200 flex flex-col overflow-hidden">
            <div className="flex-shrink-0 flex items-center justify-between p-3 border-b border-gray-200">
                <h2 className="text-base font-bold text-gray-800 flex items-center space-x-2">
                    <Video className="w-4 h-4" />
                    <span>Your Swing Video</span>
                </h2>
                <button
                    onClick={onReset}
                    className="text-sm text-gray-600 hover:text-gray-800 flex items-center space-x-1"
                >
                    <X className="w-4 h-4" />
                    <span>Clear</span>
                </button>
            </div>

            <div className="flex-1 flex flex-col p-3 overflow-y-auto">
                <video
                    src={videoUrl}
                    controls
                    className="w-full rounded-lg mb-2 flex-shrink-0"
                    style={{ maxHeight: 'calc(50vh - 150px)' }}
                >
                    Your browser does not support the video tag.
                </video>

                <div className="text-xs text-gray-600 mb-2 flex-shrink-0">
                    <p className="font-medium truncate">{file.name}</p>
                    <p className="text-xs text-gray-500">
                        {(file.size / (1024 * 1024)).toFixed(2)} MB
                    </p>
                </div>

                <button
                    onClick={onAnalyze}
                    disabled={loading}
                    className={`w-full px-4 py-2 rounded-lg font-semibold transition flex items-center justify-center space-x-2 flex-shrink-0 text-sm ${results
                            ? 'bg-green-600 hover:bg-green-700 text-white'
                            : 'bg-blue-600 hover:bg-blue-700 text-white'
                        } disabled:bg-gray-400 disabled:cursor-not-allowed`}
                >
                    {loading ? (
                        <>
                            <Activity className="w-4 h-4 animate-spin" />
                            <span>{results ? 'Re-analyzing...' : 'Analyzing...'}</span>
                        </>
                    ) : (
                        <>
                            <Activity className="w-4 h-4" />
                            <span>{results ? 'Analyze Again' : 'Analyze Swing'}</span>
                        </>
                    )}
                </button>

                {error && (
                    <div className="flex items-center space-x-2 text-red-600 bg-red-50 px-3 py-2 rounded mt-2 flex-shrink-0">
                        <AlertCircle className="w-4 h-4" />
                        <span className="text-xs">{error}</span>
                    </div>
                )}
            </div>
        </div>
    );
};