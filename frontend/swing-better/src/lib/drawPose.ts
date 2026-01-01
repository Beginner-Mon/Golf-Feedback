import { COCO_CONNECTIONS } from './constants';

export function drawPose(
    canvas: HTMLCanvasElement,
    imageBase64: string,
    joints?: number[][],
    showJoints = true
) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const img = new Image();
    img.onload = () => {
        canvas.width = img.width;
        canvas.height = img.height;
        ctx.drawImage(img, 0, 0);

        if (!showJoints || !joints) return;

        ctx.strokeStyle = '#00FF00';
        ctx.lineWidth = 3;

        COCO_CONNECTIONS.forEach(([s, e]) => {
            const [x1, y1] = joints[s];
            const [x2, y2] = joints[e];
            if (x1 > 0 && y1 > 0 && x2 > 0 && y2 > 0) {
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                ctx.stroke();
            }
        });

        ctx.fillStyle = '#FF0000';
        joints.forEach(([x, y]) => {
            if (x > 0 && y > 0) {
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, Math.PI * 2);
                ctx.fill();
            }
        });
    };

    img.src = `data:image/jpeg;base64,${imageBase64}`;
}
