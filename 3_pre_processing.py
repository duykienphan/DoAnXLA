import cv2
import os

# Thư mục đầu vào và đầu ra
input_folder = "D:\\temp\\100000_aug"
output_folder = "D:\\temp\\100000_pre"

# Tạo thư mục đầu ra nếu chưa tồn tại
os.makedirs(output_folder, exist_ok=True)

# Đếm số thứ tự file
file_count = 1

# Duyệt qua tất cả các tệp trong thư mục đầu vào
for filename in os.listdir(input_folder):
    img_path = os.path.join(input_folder, filename)
    
    # Kiểm tra định dạng ảnh (lọc tệp không hợp lệ)
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):
        print(f"Bỏ qua tệp không phải ảnh: {filename}")
        continue
    
    # Đọc ảnh
    img = cv2.imread(img_path)
    if img is None:
        print(f"Không thể đọc ảnh: {filename}")
        continue
    
    # Resize ảnh về kích thước (224, 224)
    try:
        resized_img = cv2.resize(img, (224, 224))
        
        # Tạo tên file mới theo thứ tự
        new_filename = f"{file_count:04d}.jpg"  # Định dạng 0001, 0002,...
        output_path = os.path.join(output_folder, new_filename)
        
        # Ghi ảnh với tên mới
        cv2.imwrite(output_path, resized_img)
        print(f"Đã xử lý và lưu: {new_filename}")
        
        # Tăng bộ đếm
        file_count += 1
    except Exception as e:
        print(f"Lỗi khi xử lý {filename}: {e}")
