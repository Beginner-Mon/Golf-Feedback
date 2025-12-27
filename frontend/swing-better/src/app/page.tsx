// app/page.tsx
'use client';

import React, { useState } from 'react';
import { Upload, Activity, CheckCircle, AlertCircle, TrendingUp, TrendingDown } from 'lucide-react';

interface MetricData {
  current: number;
  ideal: number;
  delta: number;
  feedback: string;
}

interface EventData {
  event: string;
  metrics: Record<string, MetricData>;
}

type Results = Record<string, EventData>;

export default function SwingAnalyzer() {
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);

  const eventNames = [
    'Address',
    'Toe-up',
    'Mid-backswing',
    'Top',
    'Mid-downswing',
    'Impact',
    'Mid-follow-through',
    'Finish'
  ];

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile && selectedFile.type === 'video/mp4') {
      setFile(selectedFile);
      setError(null);
      setResults(null);
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

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error('Analysis failed');
      }

      const data = await response.json();
      setResults(data.results);
    } catch (err) {
      setError('Failed to analyze video. Please try again.');
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const getMetricColor = (delta: number) => {
    const absDelta = Math.abs(delta);
    if (absDelta < 5) return 'text-green-600';
    if (absDelta < 15) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getMetricIcon = (delta: number) => {
    if (delta > 0) return <TrendingUp className="w-4 h-4" />;
    if (delta < 0) return <TrendingDown className="w-4 h-4" />;
    return <CheckCircle className="w-4 h-4" />;
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 p-8">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Golf Swing Analyzer
          </h1>
          <p className="text-gray-600">
            Upload your swing video for AI-powered analysis
          </p>
        </div>

        <div className="bg-white rounded-lg shadow-lg p-8 mb-8">
          <div className="flex flex-col items-center space-y-4">
            <label className="w-full max-w-md">
              <div className="flex items-center justify-center w-full h-32 px-4 transition bg-white border-2 border-gray-300 border-dashed rounded-lg hover:border-gray-400 cursor-pointer">
                <div className="flex flex-col items-center space-y-2">
                  <Upload className="w-8 h-8 text-gray-400" />
                  <span className="text-sm text-gray-600">
                    {file ? file.name : 'Click to upload MP4 video'}
                  </span>
                </div>
              </div>
              <input
                type="file"
                className="hidden"
                accept="video/mp4"
                onChange={handleFileChange}
              />
            </label>

            <button
              onClick={handleAnalyze}
              disabled={!file || loading}
              className="px-8 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed transition flex items-center space-x-2"
            >
              {loading ? (
                <>
                  <Activity className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <>
                  <Activity className="w-5 h-5" />
                  <span>Analyze Swing</span>
                </>
              )}
            </button>

            {error && (
              <div className="flex items-center space-x-2 text-red-600 bg-red-50 px-4 py-2 rounded">
                <AlertCircle className="w-5 h-5" />
                <span>{error}</span>
              </div>
            )}
          </div>
        </div>

        {results && (
          <div className="space-y-6">
            <h2 className="text-2xl font-bold text-gray-800 flex items-center space-x-2">
              <CheckCircle className="w-6 h-6 text-green-600" />
              <span>Analysis Results</span>
            </h2>

            <div className="grid gap-6">
              {Object.entries(results).map(([eventIdx, eventData]) => (
                <div
                  key={eventIdx}
                  className="bg-white rounded-lg shadow-md p-6"
                >
                  <h3 className="text-xl font-semibold text-gray-800 mb-4 border-b pb-2">
                    {eventNames[parseInt(eventIdx)] || eventData.event}
                  </h3>

                  {Object.keys(eventData.metrics).length === 0 ? (
                    <p className="text-gray-500 italic">No metrics available for this event</p>
                  ) : (
                    <div className="grid md:grid-cols-2 gap-4">
                      {Object.entries(eventData.metrics).map(([metricName, metricData]) => (
                        <div
                          key={metricName}
                          className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition"
                        >
                          <div className="flex items-center justify-between mb-3">
                            <h4 className="font-semibold text-gray-700">
                              {metricName.replace(/_/g, ' ').toUpperCase()}
                            </h4>
                            <span className={getMetricColor(metricData.delta)}>
                              {getMetricIcon(metricData.delta)}
                            </span>
                          </div>

                          <div className="space-y-2 text-sm">
                            <div className="flex justify-between">
                              <span className="text-gray-600">Current:</span>
                              <span className="font-medium">{metricData.current}°</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Ideal:</span>
                              <span className="font-medium">{metricData.ideal}°</span>
                            </div>
                            <div className="flex justify-between">
                              <span className="text-gray-600">Delta:</span>
                              <span className={`font-bold ${getMetricColor(metricData.delta)}`}>
                                {metricData.delta > 0 ? '+' : ''}{metricData.delta}°
                              </span>
                            </div>
                          </div>

                          {metricData.feedback && (
                            <div className="mt-3 pt-3 border-t border-gray-100">
                              <p className="text-sm text-gray-700 italic">
                                {metricData.feedback}
                              </p>
                            </div>
                          )}
                        </div>
                      ))}
                    </div>
                  )}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}