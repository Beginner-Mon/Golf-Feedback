// utils/index.ts
export const getMetricColor = (delta: number): string => {
    const absDelta = Math.abs(delta);
    if (absDelta < 5) return 'text-green-600';
    if (absDelta < 15) return 'text-yellow-600';
    return 'text-red-600';
};

export const getFeedbackColor = (feedback: string): string => {
    if (feedback === 'GOOD' || feedback === 'EXCELLENT') {
        return 'bg-green-100 text-green-800 border-green-300';
    }
    if (feedback === 'TOO_HIGH' || feedback === 'TOO_LOW') {
        return 'bg-red-100 text-red-800 border-red-300';
    }
    return 'bg-yellow-100 text-yellow-800 border-yellow-300';
};