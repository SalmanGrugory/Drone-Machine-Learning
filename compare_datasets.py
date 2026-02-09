from PIL import Image
import numpy as np

llvip = Image.open("C:/Users/shabd/Documents/AURORA/dataset/train/person/llvip_010001.jpg")
visdrone = Image.open("C:/Users/shabd/Documents/AURORA/dataset/train/no_person/0000007_05999_d_0000038.jpg")

llvip_arr = np.array(llvip)
visdrone_arr = np.array(visdrone)

print("LLVIP (person class):")
print(f"  Size: {llvip.size}, Mode: {llvip.mode}")
print(f"  Mean: {llvip_arr.mean():.1f}, Std: {llvip_arr.std():.1f}")
print(f"  Min: {llvip_arr.min()}, Max: {llvip_arr.max()}")

if len(llvip_arr.shape) == 3:
    r, g, b = llvip_arr[:,:,0], llvip_arr[:,:,1], llvip_arr[:,:,2]
    if np.array_equal(r, g) and np.array_equal(g, b):
        print("  Type: GRAYSCALE (R=G=B)")
    else:
        print(f"  Type: RGB (R mean={r.mean():.1f}, G mean={g.mean():.1f}, B mean={b.mean():.1f})")

print("\nVisDrone (no_person class):")
print(f"  Size: {visdrone.size}, Mode: {visdrone.mode}")
print(f"  Mean: {visdrone_arr.mean():.1f}, Std: {visdrone_arr.std():.1f}")
print(f"  Min: {visdrone_arr.min()}, Max: {visdrone_arr.max()}")

if len(visdrone_arr.shape) == 3:
    r, g, b = visdrone_arr[:,:,0], visdrone_arr[:,:,1], visdrone_arr[:,:,2]
    if np.array_equal(r, g) and np.array_equal(g, b):
        print("  Type: GRAYSCALE (R=G=B)")
    else:
        print(f"  Type: RGB (R mean={r.mean():.1f}, G mean={g.mean():.1f}, B mean={b.mean():.1f})")

print("\n** Model learned: dark/low-contrast = person, bright/high-contrast = no_person **")
