// constants/index.ts
export const COCO_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4], // Head
    [5, 6], [5, 7], [7, 9], [6, 8], [8, 10], // Arms
    [5, 11], [6, 12], [11, 12], // Torso
    [11, 13], [13, 15], [12, 14], [14, 16] // Legs
];

export const API_BASE_URL = 'http://127.0.0.1:8000';