import numpy as np

d = np.load("outputs/predicted_3d.npz", allow_pickle=True)
pred = d["positions_3d"]   # (T, 17, 3)

print("FILE: outputs/predicted_3d.npz")
print("shape:", pred.shape, "dtype:", pred.dtype)
print("min/max:", float(pred.min()), float(pred.max()))
print("mean joint norm:", float(np.mean(np.linalg.norm(pred, axis=2))))

BONES = [
    (0,1),(1,2),(2,3),
    (0,4),(4,5),(5,6),
    (0,7),(7,8),(8,9),
    (2,10),(10,11),(11,12),
    (2,13),(13,14),(14,15)
]

def mean_bone_length(seq3d, bones):
    lengths = []
    for a,b in bones:
        v = seq3d[:, b] - seq3d[:, a]
        lengths.append(np.linalg.norm(v, axis=1))
    return float(np.mean(np.stack(lengths, axis=1)))

current = mean_bone_length(pred, BONES)
target = 0.30   # meters-ish
scale = target / (current + 1e-9)

pred_scaled = pred * scale

print("Current mean bone length:", current)
print("Scale factor:", scale)

np.savez_compressed(
    "outputs/predicted_3d_denorm.npz",
    positions_3d=pred_scaled.astype(np.float32)
)