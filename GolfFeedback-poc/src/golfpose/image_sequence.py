import cv2
import os

# -------- CONFIG --------
video_path =  "du.mp4"        # Path to your MP4 file
output_dir = "golfswing/images/du"           # Folder to save frames
image_prefix = "frame"          # frame_000001.jpg
image_ext = ".jpg"
# ------------------------

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load video
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise IOError("Error opening video file")

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    filename = f"{image_prefix}_{frame_count:06d}{image_ext}"
    filepath = os.path.join(output_dir, filename)

    cv2.imwrite(filepath, frame)

cap.release()
print(f"Done! Extracted {frame_count} frames to '{output_dir}'")
