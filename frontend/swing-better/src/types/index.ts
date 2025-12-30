// types/index.ts
export interface MetricData {
    current: number;
    ideal: number;
    delta: number;
    feedback: string;
}

export interface EventMetrics {
    event: string;
    metrics: Record<string, MetricData>;
}

export interface EventFrame {
    event: string;
    image: string;
}

export interface EventJoints {
    event: string;
    joints: number[][];
}

export interface ApiResponse {
    status: string;
    event_frames?: Record<string, EventFrame>;
    joints?: Record<string, EventJoints>;
    metrics?: Record<string, EventMetrics>;
}