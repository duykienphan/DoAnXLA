import cv2
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
from playsound import playsound

# Danh mục mệnh giá tiền
categories = ['100k', '10k', '20k', '2k', '50k', '5k']
model_path = "money_final_model.h5"

# Tải mô hình
try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
    playsound("mp3/begin.mp3")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Hàm phân loại hình ảnh
def classify_image(img):
    try:
        img_resized = img.resize((224, 224), Image.Resampling.LANCZOS)
        x = image.img_to_array(img_resized)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        pred = model.predict(x, verbose=0)
        category_val = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred) * 100
        return categories[category_val], confidence
    except Exception as e:
        print(f"Error during image classification: {e}")
        return None, 0

# Biến điều khiển
money_lst = []
count = 0
result_filter = ""
prev_result_filter = ""
prev_time = time.time()
start_time = None
sound_activate = False
confidence = 0

# Khởi động webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not access the webcam.")
    exit()

print("Press 'q' to quit the application.")
while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image")
        break

    # Lật khung hình để hiển thị như gương
    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 70, 70, 560, 320
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(frame, "Detection bounding box", (190, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # Cắt vùng chứa hình ảnh cần nhận diện
    cropped_frame = frame[y1:y2, x1:x2]
    frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    # Phân loại mỗi giây
    current_time = time.time()
    if current_time - prev_time >= 1:
        result, confidence = classify_image(pil_img)
        print(f"Result: {result}, Confidence: {confidence:.2f}%")

        if result and confidence > 80:
            money_lst.append(result)
            count += 1
        else:
            money_lst.append("None")
            count += 1

        if count >= 10:
            money_frequency = {item: money_lst.count(item) for item in set(money_lst)}
            result_filter = max(money_frequency, key=money_frequency.get)
            money_lst = []
            count = 0

        prev_time = current_time
        print(money_lst)

    # Hiển thị kết quả nhận diện
    if (result_filter != "None") and (confidence > 80):
        if current_time - (start_time or current_time) <= 3:
            text1 = f"Prediction: {result_filter}"
            text2 = f"Confidence: {confidence:.2f}%"
            cv2.putText(frame, text1, (50, 350), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, text2, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if not sound_activate:
                sound_path = f"mp3/{result_filter}.mp3"
                try:
                    playsound(sound_path)
                    sound_activate = True
                except Exception as e:
                    print(f"Error playing sound: {e}")
        else:
            start_time = None  # Reset thời gian bắt đầu
    else:
        start_time = None  # Reset nếu không đủ confidence
        sound_activate = False  # Cho phép phát lại âm thanh

    # Cập nhật khi kết quả thay đổi
    if result_filter != prev_result_filter:
        sound_activate = False  # Reset cờ âm thanh cho kết quả mới
        prev_result_filter = result_filter  # Cập nhật kết quả trước đó

    # Hiển thị khung hình
    cv2.imshow("Money Classification", frame)

    # Thoát chương trình
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()
