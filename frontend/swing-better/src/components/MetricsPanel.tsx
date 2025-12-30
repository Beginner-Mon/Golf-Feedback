// components/MetricsPanel.tsx
import React from 'react';
import { CheckCircle, Activity, TrendingUp, TrendingDown } from 'lucide-react';
import { ApiResponse } from '@/types';
import { getMetricColor, getFeedbackColor } from '@/utils';

interface MetricsPanelProps {
    results: ApiResponse | null;
    loading: boolean;
}

const getMetricIcon = (delta: number) => {
    if (delta > 0) return <TrendingUp className="w-4 h-4" />;
    if (delta < 0) return <TrendingDown className="w-4 h-4" />;
    return <CheckCircle className="w-4 h-4" />;
};

export const MetricsPanel: React.FC<MetricsPanelProps> = ({ results, loading }) => {
    return (
        <div className="bg-gray-50 flex flex-col overflow-hidden">
            <div className="flex-shrink-0 p-3 bg-white border-b border-gray-200">
                <div className="flex items-center space-x-2">
                    {results?.metrics ? (
                        <>
                            <CheckCircle className="w-4 h-4 text-green-600" />
                            <h2 className="text-base font-bold text-gray-800">Analysis Results</h2>
                        </>
                    ) : (
                        <>
                            <Activity className="w-4 h-4 text-gray-400" />
                            <h2 className="text-base font-bold text-gray-800">Waiting for Analysis</h2>
                        </>
                    )}
                </div>
            </div>

            <div className="flex-1 overflow-y-auto p-3">
                {!results?.metrics ? (
                    <div className="flex items-center justify-center h-full">
                        <div className="text-center">
                            <Activity className="w-10 h-10 text-gray-300 mx-auto mb-2" />
                            <p className="text-sm text-gray-500">
                                {loading ? 'Analyzing your swing...' : 'Click "Analyze Swing" to get feedback'}
                            </p>
                        </div>
                    </div>
                ) : (
                    <div className="space-y-3">
                        {Object.entries(results.metrics)
                            .filter(([_, eventData]) => Object.keys(eventData.metrics).length > 0)
                            .map(([eventIdx, eventData]) => (
                                <div
                                    key={eventIdx}
                                    className="bg-white rounded-lg shadow-sm p-3"
                                >
                                    <h3 className="text-sm font-semibold text-gray-800 mb-2 pb-2 border-b">
                                        {eventData.event}
                                    </h3>

                                    <div className="space-y-2">
                                        {Object.entries(eventData.metrics).map(([metricName, metricData]) => (
                                            <div
                                                key={metricName}
                                                className="border border-gray-200 rounded p-2"
                                            >
                                                <div className="flex items-center justify-between mb-1">
                                                    <h4 className="font-semibold text-xs text-gray-700">
                                                        {metricName}
                                                    </h4>
                                                    <span className={getMetricColor(metricData.delta)}>
                                                        {getMetricIcon(metricData.delta)}
                                                    </span>
                                                </div>

                                                <div className="grid grid-cols-3 gap-2 text-xs mb-1">
                                                    <div>
                                                        <span className="text-gray-600 block text-xs">Current</span>
                                                        <span className="font-medium">{metricData.current}°</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-600 block text-xs">Ideal</span>
                                                        <span className="font-medium">{metricData.ideal}°</span>
                                                    </div>
                                                    <div>
                                                        <span className="text-gray-600 block text-xs">Delta</span>
                                                        <span className={`font-bold ${getMetricColor(metricData.delta)}`}>
                                                            {metricData.delta > 0 ? '+' : ''}{metricData.delta}°
                                                        </span>
                                                    </div>
                                                </div>

                                                {metricData.feedback && (
                                                    <span className={`text-xs font-semibold px-2 py-1 rounded-full border ${getFeedbackColor(metricData.feedback)}`}>
                                                        {metricData.feedback.replace(/_/g, ' ')}
                                                    </span>
                                                )}
                                            </div>
                                        ))}
                                    </div>
                                </div>
                            ))}
                    </div>
                )}
            </div>
        </div>
    );
};