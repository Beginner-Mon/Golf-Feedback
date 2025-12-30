// app/page.tsx
'use client';

import React, { useState, useRef } from 'react';
import { ApiResponse } from '@/types';
import { analyzeSwing } from '@/services/api';
import { useCanvasRenderer } from '@/hooks/useCanvasRenderer';
import { VideoUpload } from '@/components/VideoUpload';
import { VideoPlayer } from '@/components/VideoPlayer';
import { MetricsPanel } from '@/components/MetricsPanel';
import { EventFrames } from '@/components/EventFrames';

export default function SwingAnalyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<ApiResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [showJoints, setShowJoints] = useState(true);
  const [selectedFrameIdx, setSelectedFrameIdx] = useState<string | null>(null);
  const canvasRefs = useRef<Record<string, HTMLCanvasElement | null>>({});

  useCanvasRenderer(results, showJoints, canvasRefs);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && selectedFile.type === 'video/mp4') {
      setFile(selectedFile);
      setError(null);

      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
      const url = URL.createObjectURL(selectedFile);
      setVideoUrl(url);
    } else {
      setError('Please select a valid MP4 video file');
    }
  };

  const handleAnalyze = async () => {
    if (!file) {
      setError('Please select a video file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const data = await analyzeSwing(file);
      setResults(data);
      if (data.event_frames) {
        setSelectedFrameIdx(Object.keys(data.event_frames)[0]);
      }
    } catch (err) {
      setError('Failed to analyze video. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setVideoUrl(null);
    setResults(null);
    setError(null);
    setSelectedFrameIdx(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">

      {/* ===================== */}
      {/* SECTION 1: ANALYZER   */}
      {/* ===================== */}
      <div className="h-screen flex flex-col overflow-hidden">

        {/* Header */}
        <div className="flex-shrink-0 text-center py-4 px-4 border-b border-gray-200 bg-white/50 backdrop-blur">
          <h1 className="text-2xl font-bold text-gray-800 mb-1">
            Golf Swing Analyzer
          </h1>
          <p className="text-xs text-gray-600">
            Upload your swing video for AI-powered analysis
          </p>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-hidden">
          {!videoUrl ? (
            <VideoUpload onFileChange={handleFileChange} error={error} />
          ) : (
            <div className="h-full grid lg:grid-cols-2">
              <VideoPlayer
                videoUrl={videoUrl}
                file={file!}
                loading={loading}
                results={results}
                error={error}
                onAnalyze={handleAnalyze}
                onReset={handleReset}
              />
              <MetricsPanel results={results} loading={loading} />
            </div>
          )}
        </div>
      </div>

      {/* ===================== */}
      {/* SECTION 2: EVENTS     */}
      {/* ===================== */}

      {results?.event_frames && (
        <EventFrames
          results={results}
          showJoints={showJoints}
          selectedFrameIdx={selectedFrameIdx}
          canvasRefs={canvasRefs}
          onToggleJoints={() => setShowJoints(!showJoints)}
          onSelectFrame={setSelectedFrameIdx}
        />
      )}


    </div>


  );
}