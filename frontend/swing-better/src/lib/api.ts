import { ApiResponse } from '@/types';

export async function analyzeSwing(file: File): Promise<ApiResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('return_frames', 'true');
    formData.append('return_joints', 'true');
    formData.append('return_metrics', 'true');

    const res = await fetch('http://127.0.0.1:8000/analyze/components', {
        method: 'POST',
        body: formData,
    });

    if (!res.ok) throw new Error('Analysis failed');

    return res.json();
}
