import os
from PIL import Image

# Đường dẫn đến thư mục chứa dataset
dataset_folder = 'datasets/MSD-US/test/oil' # Thay thế đường dẫn phù hợp

# Khởi tạo danh sách để lưu kích thước ảnh
image_sizes = []

# Duyệt qua tất cả các tệp trong thư mục dataset
for root, dirs, files in os.walk(dataset_folder):
    for file in files:
        if file.endswith((".jpg", ".jpeg", ".png", ".bmp", ".tiff")):  # Kiểm tra nếu file là ảnh
            image_path = os.path.join(root, file)
            try:
                # Mở ảnh và lấy kích thước
                with Image.open(image_path) as img:
                    image_sizes.append(img.size)
            except Exception as e:
                print(f"Không thể mở ảnh {image_path}: {e}")

# Kiểm tra kích thước của tất cả các ảnh
if len(image_sizes) == 0:
    print("Không có ảnh nào được tìm thấy trong thư mục.")
else:
    # Tạo tập hợp (set) để loại bỏ các kích thước trùng lặp
    unique_sizes = set(image_sizes)

    # In ra kết quả
    if len(unique_sizes) == 1:
        print(f"Tất cả các ảnh đều có cùng kích thước: {unique_sizes.pop()}")
    else:
        print(f"Có nhiều kích thước ảnh khác nhau trong dataset:")
        for size in unique_sizes:
            print(f"- Kích thước: {size}")