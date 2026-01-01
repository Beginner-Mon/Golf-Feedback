import numpy as np

NPZ_PATH = "predicted_3d_denorm.npz"  # adjust path if needed


def inspect_npz(path: str):
    print(f"Loading NPZ file: {path}\n")

    data = np.load(path, allow_pickle=True)

    print("Keys found in NPZ:")
    for key in data.files:
        print(f" - {key}")

    print("\nDetailed inspection:\n")

    for key in data.files:
        arr = data[key]
        print("=" * 60)
        print(f"Key: {key}")
        print(f"Type: {type(arr)}")

        if isinstance(arr, np.ndarray):
            print(f"Shape: {arr.shape}")
            print(f"Dtype: {arr.dtype}")

            # Show a small preview safely
            try:
                if arr.ndim == 0:
                    print("Value:", arr.item())
                elif arr.ndim == 1:
                    print("First 10 values:", arr[:10])
                elif arr.ndim >= 2:
                    print("First element slice:")
                    print(arr[0])
            except Exception as e:
                print("Preview error:", e)
        else:
            print("Value:", arr)

        print()

    data.close()


if __name__ == "__main__":
    inspect_npz(NPZ_PATH)
