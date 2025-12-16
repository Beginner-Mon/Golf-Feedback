import cv2
import os


def preprocess_video(
    input_video_path,
    output_video_path,
    bbox,
    events,
    dim=160
):
    """
    Preprocess a video by cropping, resizing, padding, and saving.
    
    Args:
        input_video_path (str): Path to original video.
        output_video_path (str): Where to save processed video.
        bbox (list|tuple): [x, y, w, h] in normalized coordinates (0–1).
        events (list|tuple): [start_frame, ..., end_frame].
        dim (int): Output video dimension (square).
    """

    # Open video
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Cannot open video: {input_video_path}")
        return
    
    # Output writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (dim, dim))

    # Convert normalized bbox to absolute pixel values
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    x = int(frame_w * bbox[0])
    y = int(frame_h * bbox[1])
    w = int(frame_w * bbox[2])
    h = int(frame_h * bbox[3])

    start_frame = events[0]
    end_frame = events[-1]

    count = 0
    success, frame = cap.read()

    print("Processing...")

    while success:
        count += 1

        # Only process frames inside event window
        if start_frame <= count <= end_frame:

            crop = frame[y:y + h, x:x + w]
            crop_h, crop_w = crop.shape[:2]

            # Resize with aspect ratio
            scale = dim / max(crop_h, crop_w)
            new_w = int(crop_w * scale)
            new_h = int(crop_h * scale)
            resized = cv2.resize(crop, (new_w, new_h))

            # Pad to square with ImageNet mean colors (BGR)
            delta_w = dim - new_w
            delta_h = dim - new_h
            top, bottom = delta_h // 2, delta_h - (delta_h // 2)
            left, right = delta_w // 2, delta_w - (delta_w // 2)

            pad_color = [0.406 * 255, 0.456 * 255, 0.485 * 255]
            final_frame = cv2.copyMakeBorder(
                resized, top, bottom, left, right,
                cv2.BORDER_CONSTANT, value=pad_color
            )

            out.write(final_frame)

        # Stop early
        if count > end_frame:
            break

        success, frame = cap.read()

    cap.release()
    out.release()
    print("Done! Saved to:", output_video_path)


if __name__ == "__main__":
    # ---- Example usage ----

    input_file = "../test_video.mp4"
    output_file = "processed_video1.mp4"

    # Normalized bbox: [x, y, w, h]
    bbox = [0.3, 0.2, 0.4, 0.6]

    # Event frames
    events = [100, 150, 200]

    preprocess_video(input_file, output_file, bbox, events, dim=160)
