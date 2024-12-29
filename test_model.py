# pip install tensorflow==2.15.0

import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from PIL import Image
import numpy as np
import tensorflow as tf
import cv2

# Lấy danh sách class
categories = os.listdir("D:\\KienPhan\\Projects\\money_detection_rpi4\\images\\dataset_for_model\\train")
categories.sort()  # Sắp xếp danh sách class theo thứ tự
print(f"Categories: {categories}")

# Load mô hình đã lưu
model_path = "D:\\KienPhan\\Projects\\money_detection_rpi4\\money_model_best.h5"
model = tf.keras.models.load_model(model_path)

print("Model loaded successfully!")
print(model.summary())

# Hàm phân loại ảnh
def classify_image(img_file):
    try:
        # Mở và resize ảnh
        img = Image.open(img_file)
        img = img.resize((224, 224), Image.Resampling.LANCZOS)

        # Chuyển đổi ảnh thành mảng
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        print(f"Input shape for model: {x.shape}")

        # Dự đoán
        pred = model.predict(x)
        print(f"Prediction probabilities: {pred}")

        # Lấy class có xác suất cao nhất
        category_val = np.argmax(pred, axis=1)[0]
        print(f"Predicted class index: {category_val}")

        result = categories[category_val]
        return result

    except Exception as e:
        print(f"Error processing image {img_file}: {e}")
        return None

# Đường dẫn ảnh test
image_path = "D:\\KienPhan\\Projects\\money_detection_rpi4\\images\\dataset_for_model\\test\\test1.jpg"

# Phân loại ảnh
result_txt = classify_image(image_path)
print(f"Predicted category: {result_txt}")

img = cv2.imread(image_path)
img = cv2.putText(img, result_txt, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()