// components/VideoUpload.tsx
import React from 'react';
import { Upload, AlertCircle } from 'lucide-react';

interface VideoUploadProps {
    onFileChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
    error: string | null;
}

export const VideoUpload: React.FC<VideoUploadProps> = ({ onFileChange, error }) => {
    return (
        <div className="flex-1 flex items-center justify-center p-4">
            <div className="bg-white rounded-lg shadow-lg p-8 max-w-2xl w-full">
                <div className="flex flex-col items-center space-y-4">
                    <label className="w-full">
                        <div className="flex items-center justify-center w-full h-64 px-4 transition bg-white border-2 border-gray-300 border-dashed rounded-lg hover:border-gray-400 cursor-pointer">
                            <div className="flex flex-col items-center space-y-2">
                                <Upload className="w-16 h-16 text-gray-400" />
                                <span className="text-lg text-gray-600 text-center">
                                    Click to upload MP4 video
                                </span>
                            </div>
                        </div>
                        <input
                            type="file"
                            className="hidden"
                            accept="video/mp4"
                            onChange={onFileChange}
                        />
                    </label>

                    {error && (
                        <div className="flex items-center space-x-2 text-red-600 bg-red-50 px-4 py-2 rounded w-full">
                            <AlertCircle className="w-5 h-5" />
                            <span>{error}</span>
                        </div>
                    )}
                </div>
            </div>
        </div>
    );
};