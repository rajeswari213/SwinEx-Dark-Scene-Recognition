import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import pickle

# ===============================
# DATASET PATH
# ===============================
dataset_path = r".\dataset"   # change to your dataset path

classes = os.listdir(dataset_path)

brightness_values = []
noise_values = []
widths = []
heights = []
class_counts = {}

# ===============================
# RETINEX ENHANCEMENT FUNCTION
# ===============================
def retinex_enhancement(img):

    img = img.astype(np.float32) + 1.0
    log_img = np.log(img)

    blur = cv2.GaussianBlur(img,(31,31),0)
    log_blur = np.log(blur)

    retinex = log_img - log_blur
    retinex = cv2.normalize(retinex,None,0,255,cv2.NORM_MINMAX)
    retinex = np.uint8(retinex)

    return retinex

# ===============================
# NOISE ESTIMATION FUNCTION
# ===============================
def estimate_noise(image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    noise = cv2.Laplacian(gray, cv2.CV_64F).var()
    return noise


# ===============================
# READ DATASET
# ===============================
for cls in classes:

    image_paths = glob(os.path.join(dataset_path,cls,"*.jpg"))
    image_paths += glob(os.path.join(dataset_path,cls,"*.png"))

    class_counts[cls] = len(image_paths)

    for img_path in image_paths:

        img = cv2.imread(img_path)

        if img is None:
            continue

        h,w,_ = img.shape

        widths.append(w)
        heights.append(h)

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        brightness = np.mean(gray)
        brightness_values.append(brightness)

        noise = estimate_noise(img)
        noise_values.append(noise)

# ===============================
# PLOT 5: BEFORE vs AFTER ENHANCEMENT
# ===============================
sample_image = glob(os.path.join(dataset_path,classes[0],"*"))[0]

img = cv2.imread(sample_image)
enhanced = retinex_enhancement(img)


# ===============================
# SHOW SAMPLE PREPROCESSING
# ===============================
plt.figure()

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
plt.title("Original Dark Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(enhanced,cv2.COLOR_BGR2RGB))
plt.title("SwinEx Preprocessing")
plt.axis("off")

plt.show()

# ======================================================
# Load pickle file
# ======================================================
with open("exdark_metrics.pkl", "rb") as f:
    data = pickle.load(f)

epochs = data["epochs"]

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["miou"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("mIoU", fontsize=13, fontweight='bold')
plt.title("mIoU Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')
plt.grid(True)

plt.show()

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["psnr"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("PSNR", fontsize=13, fontweight='bold')
plt.title("PSNR Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')
plt.grid(True)

plt.show()

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["ssim"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("SSIM", fontsize=13, fontweight='bold')
plt.title("SSIM Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.xticks(fontsize=11, fontweight='bold')
plt.yticks(fontsize=11, fontweight='bold')
plt.grid(True)

plt.show()

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["fsim"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("FSIM", fontsize=13, fontweight='bold')
plt.title("FSIM Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["mae"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("MAE", fontsize=13, fontweight='bold')
plt.title("MAE Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["gmsd"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("GMSD", fontsize=13, fontweight='bold')
plt.title("GMSD Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["sam"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("SAM", fontsize=13, fontweight='bold')
plt.title("SAM Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["srer"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("SRER", fontsize=13, fontweight='bold')
plt.title("SRER Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["uqi"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("UQI", fontsize=13, fontweight='bold')
plt.title("UQI Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.grid(True)
plt.show()

plt.figure(figsize=(8,5))
plt.plot(data["epochs"], data["rmse"], marker='o')

plt.xlabel("Epochs", fontsize=13, fontweight='bold')
plt.ylabel("RMSE", fontsize=13, fontweight='bold')
plt.title("RMSE Performance with EX-DARK Dataset", fontsize=16, fontweight='bold')

plt.grid(True)
plt.show()