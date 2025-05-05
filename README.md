# Hệ thống Nhận diện và Tìm kiếm Khuôn mặt

Ứng dụng web cho phép xây dựng cơ sở dữ liệu lưu trữ và tìm kiếm khuôn mặt người dựa trên các đặc trưng như cảm xúc, tuổi và màu da.

![Demo Screenshot](screenshot.png)

## Tính năng chính

1. **Xây dựng Cơ sở dữ liệu**
   - Trích xuất đặc trưng khuôn mặt từ bộ dữ liệu ảnh
   - Nhận diện cảm xúc, ước tính tuổi và màu da
   - Lưu trữ các đặc trưng để tìm kiếm

2. **Tìm kiếm Khuôn mặt**
   - Tải lên ảnh khuôn mặt mới
   - Tìm kiếm 3 khuôn mặt tương tự nhất trong CSDL
   - Sắp xếp theo độ tương đồng giảm dần

3. **Lọc Khuôn mặt**
   - Lọc khuôn mặt theo cảm xúc
   - Lọc khuôn mặt theo độ tuổi

## Kiến trúc hệ thống

![System Architecture](architecture.png)

### Các thành phần chính
- **Frontend**: HTML/CSS/JavaScript với Bootstrap 5
- **Backend**: Python Flask
- **Xử lý ảnh**: OpenCV, face_recognition
- **Phân tích khuôn mặt**: DeepFace

## Cài đặt

### Yêu cầu hệ thống
- Python 3.7+
- pip

### Cài đặt thư viện
```bash
pip install -r requirements.txt
```
> Lưu ý: Thư viện dlib có thể cần cài đặt thêm các gói phụ thuộc như CMake và C++ compiler.

### Cấu trúc thư mục
```
├── app.py                  # Mã nguồn chính
├── requirements.txt        # Các thư viện phụ thuộc
├── templates/              # Thư mục chứa template HTML
│   └── index.html          # Giao diện người dùng
├── data_test/              # Thư mục chứa dữ liệu ảnh khuôn mặt
└── uploads/                # Thư mục lưu trữ ảnh tải lên
```

## Cách sử dụng

1. Khởi động ứng dụng:
```bash
python app.py
```

2. Truy cập ứng dụng qua trình duyệt web:
```
http://localhost:5000
```

3. Xây dựng cơ sở dữ liệu:
   - Nhấn nút "Xây dựng CSDL" để bắt đầu phân tích và trích xuất đặc trưng từ các ảnh trong thư mục `data_test`

4. Tìm kiếm khuôn mặt:
   - Tải lên ảnh khuôn mặt cần tìm
   - Xem kết quả hiển thị 3 khuôn mặt tương tự nhất

5. Lọc khuôn mặt:
   - Chọn cảm xúc và/hoặc khoảng tuổi để lọc
   - Xem kết quả hiển thị các khuôn mặt phù hợp với tiêu chí

## Giải thích thuật toán

### Trích xuất đặc trưng
- **Cảm xúc**: Sử dụng DeepFace để phân loại cảm xúc thành 7 loại (tức giận, ghê tởm, sợ hãi, vui vẻ, buồn bã, ngạc nhiên, trung tính)
- **Tuổi**: Sử dụng DeepFace để ước tính tuổi dựa trên mô hình học máy đã được huấn luyện
- **Màu da**: Trích xuất giá trị màu trung bình trong vùng khuôn mặt ở không gian màu HSV

### Tìm kiếm tương đồng
- Sử dụng encoding 128D từ thư viện face_recognition (dựa trên kiến trúc mạng ResNet)
- Tính toán khoảng cách Euclid giữa vector đặc trưng của ảnh đầu vào và các ảnh trong CSDL
- Sắp xếp kết quả theo độ tương đồng (1 - khoảng cách) và chọn top 3

## Tác giả

Dự án được phát triển bởi [Tên tác giả] © 2023. 