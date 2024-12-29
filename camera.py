import cv2

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = cv2.flip(frame, 1)  # Lật ngang
        cv2.rectangle(frame, (50,70), (580,320), (0, 0, 255), 3)

        if not ret:
            print("Failed to capture frame. Exiting...")
            break

        cv2.imshow('Camera', frame)

        if cv2.waitKey(1) & 0xFF == ord('c'):
            # Lưu ảnh
            flip_frame = cv2.flip(frame, 1)  # Lật ngang
            image_path = "captured_image.jpg"
            cv2.imwrite(image_path, flip_frame)
            print(f"Ảnh đã được lưu tại: {image_path}")

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
