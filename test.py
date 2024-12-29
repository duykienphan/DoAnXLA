import time

pre_time = 0

while True:
    # Lấy thời gian thực dưới dạng Unix timestamp (số giây kể từ 01/01/1970)
    current_time = int(time.time())

    if current_time - pre_time >= 1:
        print("Thời gian thực (Unix timestamp):", current_time)
    pre_time = current_time