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


export type Pose3DFrame = {
    frame: number;
    joints_3d: number[][];
};

type Pose3DResponse = {
    status: string;
    pagination: {
        page: number;
        page_size: number;
        total_frames: number;
        total_pages: number;
    };
    data: Pose3DFrame[];
};