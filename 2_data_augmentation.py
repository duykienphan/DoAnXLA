import cv2
import numpy as np
import random
import os

# Đường dẫn thư mục đầu vào và thư mục đầu ra
input_folder = "D:\\temp\\100000"
output_folder = "D:\\temp\\100000_aug"
os.makedirs(output_folder, exist_ok=True)

# Hàm thay đổi độ sáng của ảnh
def adjust_brightness(img, factor):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv = np.array(hsv, dtype=np.float64)
    hsv[:, :, 2] = hsv[:, :, 2] * factor
    hsv[:, :, 2][hsv[:, :, 2] > 255] = 255
    hsv = np.array(hsv, dtype=np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Hàm xoay ảnh
def rotate_image(img, angle):
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(img, matrix, (w, h))

# Hàm dịch chuyển ảnh
def translate_image(img, tx, ty):
    h, w = img.shape[:2]
    matrix = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv2.warpAffine(img, matrix, (w, h))

# Hàm bóp méo ảnh (perspective transformation)
def distort_image(img):
    h, w = img.shape[:2]
    pts1 = np.float32([
        [random.randint(0, w // 4), random.randint(0, h // 4)],
        [random.randint(3 * w // 4, w), random.randint(0, h // 4)],
        [random.randint(0, w // 4), random.randint(3 * h // 4, h)],
        [random.randint(3 * w // 4, w), random.randint(3 * h // 4, h)]
    ])
    pts2 = np.float32([
        [0, 0],
        [w - 1, 0],
        [0, h - 1],
        [w - 1, h - 1]
    ])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    return cv2.warpPerspective(img, matrix, (w, h))

# Lặp qua tất cả các ảnh trong thư mục đầu vào
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)

    # Kiểm tra định dạng ảnh
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Bỏ qua tệp không phải ảnh: {filename}")
        continue

    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc ảnh: {filename}")
        continue

    # Tăng cường dữ liệu
    num_augmentations = 100
    for i in range(num_augmentations):
        augmented_img = img.copy()

        # Chọn một kỹ thuật tăng cường ngẫu nhiên
        augmentation_type = random.choice(["rotate", "translate", "brightness", "distort"])

        if augmentation_type == "rotate":
            angle = random.uniform(0, 360)
            augmented_img = rotate_image(augmented_img, angle)

        elif augmentation_type == "translate":
            tx = random.randint(-30, 30)  # Dịch ngang
            ty = random.randint(-30, 30)  # Dịch dọc
            augmented_img = translate_image(augmented_img, tx, ty)

        elif augmentation_type == "brightness":
            factor = random.uniform(0.5, 1.5)  # Tăng/giảm độ sáng
            augmented_img = adjust_brightness(augmented_img, factor)

        elif augmentation_type == "distort":
            augmented_img = distort_image(augmented_img)

        # Lưu ảnh tăng cường
        base_filename, ext = os.path.splitext(filename)
        output_path = os.path.join(output_folder, f"{base_filename}_augmented_{i + 1:03d}.jpg")
        cv2.imwrite(output_path, augmented_img)

        print(f"Đã lưu: {output_path}")

print("Hoàn thành tăng cường dữ liệu!")