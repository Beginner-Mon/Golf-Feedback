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


export interface PoseFrameData {
    joints_3d: number[][];
}

export interface PaginationData {
    page: number;
    page_size: number;
    total_frames: number;
    total_pages: number;
}

export interface MetaData {
    joints: number;
    total_frames: number;
}
export interface PoseApiResponse {
    status: 'success' | 'error';
    data: PoseFrameData[];
    meta: MetaData;
    pagination: PaginationData;
}