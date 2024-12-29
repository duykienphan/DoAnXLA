import cv2, time
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image

categories = ['100k', '10k', '20k', '2k', '50k', '5k']
model_path = "D:\\KienPhan\\Projects\\money_detection_rpi4\\money_final_model.h5"

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

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

money_lst = []
count = 0
result_filter = ""
prev_time = time.time()
start_time = None
money_frequency = {}

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

    frame = cv2.flip(frame, 1)
    x1, y1, x2, y2 = 70, 70, 560, 320
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)
    cv2.putText(frame, "Detection bouding box", (190, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cropped_frame = frame[y1:y2, x1:x2]
    flip_frame = cv2.flip(cropped_frame, 1)
    frame_rgb = cv2.cvtColor(flip_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(frame_rgb)

    current_time = time.time()
    if current_time - prev_time >= 1:
        result, confidence = classify_image(pil_img)
        print(result, confidence)
        if result and confidence > 80:
            money_lst.append(result)
            count += 1
        elif result and confidence < 80:
            money_lst.append("None")
            count += 1

        if count >= 10:
            money_frequency = {}
            for item in money_lst:
                money_frequency[item] = money_frequency.get(item, 0) + 1
            #print(money_frequency)
            result_filter = max(money_frequency, key=money_frequency.get)
            money_lst = []
            count = 0

        prev_time = current_time
    
    if start_time is None:  # Nếu chưa bắt đầu hiển thị, đặt thời gian bắt đầu
        start_time = current_time

    if result_filter != "None" and confidence > 80:
        if current_time - start_time <= 3:
            text1 = f"Prediction: {result_filter}"
            text2 = f"Confidence: {confidence:.2f}%"
            cv2.putText(frame, text1, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, text2, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            start_time = None # Đã hiển thị đủ 5 giây, reset start_time để chờ lần hiển thị tiếp theo

    cv2.imshow("Money Classification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()