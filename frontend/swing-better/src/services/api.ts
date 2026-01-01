// services/api.ts
import axios from 'axios';
import { ApiResponse } from '@/types';
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