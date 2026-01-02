import numpy as np

npz_path = "golfswing/data_2d_golf_gt.npz"

data = np.load(npz_path, allow_pickle=True)
positions_2d = data["positions_2d"].item()

print("Swing phases:", positions_2d.keys())
print()

for g in ["G1", "G2", "G3", "G4", "G5", "G6"]:
    print(f"===== {g} =====")
    phase = positions_2d[g]

    vid = next(iter(phase))
    frames = phase[vid]   # THIS IS A LIST

    print("Sample video:", vid)
    print("Type:", type(frames))
    print("Number of frames:", len(frames))

    # Convert to numpy for inspection
    arr = np.asarray(frames)

    print("Converted shape:", arr.shape)
    print("Dtype:", arr.dtype)

    # Detect 2D joints
    if arr.ndim == 3 and arr.shape[-1] == 2:
        print("-> 2D GT format (frames, joints, xy)")

    print("X range:", arr[..., 0].min(), "to", arr[..., 0].max())
    print("Y range:", arr[..., 1].min(), "to", arr[..., 1].max())
    print()
