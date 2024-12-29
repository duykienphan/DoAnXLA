import cv2
import os
import time

def capture_images(num_images, capture_time=5, path="D:\\temp\\10000", start_index=0):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Couldn't access the camera.")
        return
    
    # Create the directory if it doesn't exist
    os.makedirs(path, exist_ok=True)

    for i in range(num_images):
        ret, frame = cap.read()
        if not ret:
            print("Error: Couldn't capture an image.")
            break

        # Display the captured image
        start_time = time.time()
        while time.time() - start_time < capture_time:
            remaining_time = int(capture_time - (time.time() - start_time))
            frame_with_text = frame.copy()
            cv2.putText(frame_with_text, f"Time left: {remaining_time}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow('Data collection', frame_with_text)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        image_name = os.path.join(path, f'image_{i+start_index+1}.jpg')
        cv2.imwrite(image_name, frame)
        print(f"Image {i+start_index+1} captured as {image_name}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_images(num_images=200, capture_time=3, path="D:\\temp\\5000", start_index=0)
