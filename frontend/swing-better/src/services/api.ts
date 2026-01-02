// services/api.ts
import axios from 'axios';
import { ApiResponse, PoseApiResponse } from '@/types';
import { API_BASE_URL } from '@/lib/constants';

const apiClient = axios.create({
    baseURL: API_BASE_URL,
    headers: {
        'Content-Type': 'multipart/form-data',
    },
});

export const analyzeSwing = async (file: File): Promise<ApiResponse> => {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('return_frames', 'true');
    formData.append('return_joints', 'true');
    formData.append('return_metrics', 'true');

    const response = await apiClient.post<ApiResponse>('/analyze/components', formData);
    return response.data;
};

export const fetchPoseData = async (page: number = 1, pageSize: number = 30): Promise<PoseApiResponse> => {
    const response = await axios.get<PoseApiResponse>(`${API_BASE_URL}/analyze/3d`, {
        params: { page, page_size: pageSize }
    });
    return response.data;
};