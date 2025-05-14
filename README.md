# Hệ Thống Nhận Diện Khuôn Mặt

Hệ thống nhận diện khuôn mặt trích xuất các đặc trưng từ khuôn mặt (giới tính, màu da, cảm xúc) và tìm kiếm những khuôn mặt tương tự dựa trên khoảng cách Euclidean.

## Tính Năng

- Trích xuất giới tính, màu da và cảm xúc từ hình ảnh khuôn mặt
- Tìm top 3 khuôn mặt tương tự nhất dựa trên khoảng cách Euclidean
- Lọc kết quả tìm kiếm theo giới tính, màu da và cảm xúc
- Xây dựng và quản lý cơ sở dữ liệu thông qua giao diện web
- Hỗ trợ song ngữ Anh-Việt cho mô tả đặc trưng

## Kiến Trúc Hệ Thống

Hệ thống bao gồm các thành phần sau:

- **Trích xuất đặc trưng**: Trích xuất đặc trưng giới tính, màu da và cảm xúc từ hình ảnh khuôn mặt
- **Quản lý cơ sở dữ liệu**: Lưu trữ và truy xuất đặc trưng khuôn mặt và hình ảnh
- **Giao diện Web**: Cung cấp giao diện thân thiện với người dùng để tìm kiếm và quản lý cơ sở dữ liệu

## Cài Đặt

### Yêu Cầu Hệ Thống

- Python 3.7 trở lên
- MySQL 5.7 trở lên
- Thư viện OpenCV và NumPy

### Thiết Lập

1. Sao chép kho lưu trữ:
   ```
   git clone https://github.com/username/facial-recognition.git
   cd facial-recognition
   ```

2. Cài đặt các gói phụ thuộc:
   ```
   pip install -r requirements.txt
   ```

3. Cấu hình kết nối cơ sở dữ liệu trong `db_config.py`:
   ```python
   DB_HOST = 'localhost'
   DB_USER = 'your_username'
   DB_PASSWORD = 'your_password'
   DB_NAME = 'face_recognition_db'
   DB_PORT = 3306
   ```

4. Thiết lập cơ sở dữ liệu:
   ```
   python mysql_setup.py
   ```

## Sử Dụng

1. Khởi động ứng dụng:
   ```
   python app.py
   ```

2. Mở trình duyệt web và truy cập:
   ```
   http://localhost:8080
   ```

3. Sử dụng tab "Quản lý cơ sở dữ liệu" để thêm hình ảnh vào cơ sở dữ liệu bằng cách cung cấp đường dẫn đến thư mục chứa hình ảnh khuôn mặt.

4. Sử dụng tab "Tìm kiếm" để tải lên hình ảnh khuôn mặt và tìm kiếm các khuôn mặt tương tự trong cơ sở dữ liệu.

5. Áp dụng các bộ lọc (giới tính, màu da, cảm xúc) để tinh chỉnh kết quả tìm kiếm.

## Kiểm Tra Các Module

Bạn có thể kiểm tra các module trích xuất đặc trưng riêng lẻ bằng cách sử dụng script kiểm tra được cung cấp:

```
python test_features.py đường/dẫn/đến/ảnh.jpg
```

Để so sánh đặc trưng giữa hai hình ảnh:

```
python test_features.py đường/dẫn/đến/ảnh1.jpg đường/dẫn/đến/ảnh2.jpg
```

## Mô Tả Module

- `app.py`: Ứng dụng Flask chính xử lý các route và yêu cầu
- `database.py`: Xử lý các thao tác cơ sở dữ liệu để lưu trữ và truy xuất dữ liệu khuôn mặt
- `feature_extraction.py`: Kết hợp tất cả các đặc trưng (giới tính, màu da, cảm xúc) thành một vector duy nhất
- `gender_detection.py`: Phát hiện giới tính và trích xuất đặc trưng liên quan đến giới tính
- `skin_classification.py`: Phân loại màu da và trích xuất đặc trưng màu da
- `emotion_detection.py`: Phát hiện cảm xúc và trích xuất đặc trưng liên quan đến cảm xúc
- `utils.py`: Chứa các hàm tiện ích cho dịch thuật và xác thực
- `db_config.py`: Chứa cài đặt cấu hình cơ sở dữ liệu
- `mysql_setup.py`: Thiết lập cơ sở dữ liệu MySQL với các bảng cần thiết
- `test_features.py`: Script kiểm tra để xác minh chức năng trích xuất đặc trưng

## Tìm Kiếm Top 3 Khuôn Mặt Tương Tự Nhất

### Phương Pháp Khoảng Cách Euclidean

Hệ thống sử dụng khoảng cách Euclidean để đo lường độ tương tự giữa các vector đặc trưng khuôn mặt. Công thức tính khoảng cách Euclidean giữa hai vector đặc trưng A và B được định nghĩa như sau:

```
d(A, B) = √[(a₁ - b₁)² + (a₂ - b₂)² + ... + (aₙ - bₙ)²]
```

Trong đó:
- A = (a₁, a₂, ..., aₙ) là vector đặc trưng của khuôn mặt truy vấn
- B = (b₁, b₂, ..., bₙ) là vector đặc trưng của khuôn mặt trong cơ sở dữ liệu
- n là số chiều của vector đặc trưng (trong trường hợp của chúng ta, n = 48, bao gồm 16 chiều cho giới tính + 16 chiều cho màu da + 16 chiều cho cảm xúc)

Các bước thực hiện:
1. Trích xuất vector đặc trưng 48 chiều từ hình ảnh khuôn mặt đầu vào
2. Tính toán khoảng cách Euclidean giữa vector này và tất cả các vector trong cơ sở dữ liệu
3. Sắp xếp các kết quả theo khoảng cách Euclidean tăng dần (khoảng cách càng nhỏ, độ tương tự càng cao)
4. Chọn 3 khuôn mặt có khoảng cách nhỏ nhất (top 3 tương tự nhất)

### Trọng Số Đặc Trưng

Mỗi loại đặc trưng (giới tính, màu da, cảm xúc) có thể được gán các trọng số khác nhau để điều chỉnh ảnh hưởng của chúng trong quá trình tính toán khoảng cách:

```
d(A, B) = √[w₁·∑(a₁ᵢ - b₁ᵢ)² + w₂·∑(a₂ᵢ - b₂ᵢ)² + w₃·∑(a₃ᵢ - b₃ᵢ)²]
```

Trong đó:
- w₁, w₂, w₃ là các trọng số cho đặc trưng giới tính, màu da và cảm xúc
- a₁ᵢ, a₂ᵢ, a₃ᵢ là các thành phần của vector đặc trưng A tương ứng với giới tính, màu da và cảm xúc
- b₁ᵢ, b₂ᵢ, b₃ᵢ là các thành phần của vector đặc trưng B tương ứng với giới tính, màu da và cảm xúc

## Vector Đặc Trưng 16 Chiều

Mỗi loại đặc trưng (giới tính, màu da, cảm xúc) được biểu diễn bởi một vector 16 chiều. Dưới đây là ý nghĩa chi tiết của từng chiều trong mỗi vector.

### Vector Đặc Trưng Giới Tính (16 chiều)

1. **Tỷ lệ chiều rộng/chiều cao của khuôn mặt**: Đàn ông thường có tỷ lệ này lớn hơn
2. **Mức độ cường điệu xương hàm**: Đo đặc trưng về độ góc cạnh của xương hàm
3. **Chiều dài của trán**: Đo từ đỉnh lông mày đến đường chân tóc
4. **Khoảng cách giữa hai mắt**: Đo khoảng cách tương đối giữa hai mắt
5. **Độ dày của lông mày**: Đo độ dày trung bình của lông mày
6. **Góc lông mày**: Đo góc của lông mày so với mặt phẳng ngang
7. **Tỷ lệ chiều cao/chiều rộng của mũi**: Đặc trưng về hình dạng mũi
8. **Độ rộng của cằm**: Đo độ rộng tương đối của cằm
9. **Độ nhọn của cằm**: Đo mức độ nhọn ở phần cuối cằm
10. **Tỷ lệ môi trên/môi dưới**: Tỷ lệ độ dày giữa môi trên và môi dưới
11. **Độ nổi của xương gò má**: Đo độ nổi của xương gò má
12. **Độ mịn da**: Đánh giá độ mịn của da mặt
13. **Chiều dài tương đối của cổ**: Đo độ dài tương đối của cổ
14. **Độ vòng của gò má**: Đo độ tròn của gò má
15. **Kích thước của tai**: Đo kích thước tương đối của tai
16. **Hệ số tin cậy**: Mức độ tin cậy của dự đoán giới tính

### Vector Đặc Trưng Màu Da (16 chiều)

1. **Giá trị trung bình kênh H (Hue)**: Giá trị trung bình của thông số màu sắc trong không gian màu HSV
2. **Giá trị trung bình kênh S (Saturation)**: Giá trị trung bình của độ bão hòa màu trong không gian màu HSV
3. **Giá trị trung bình kênh V (Value)**: Giá trị trung bình của độ sáng trong không gian màu HSV
4. **Độ lệch chuẩn kênh H**: Độ biến thiên của thông số màu sắc
5. **Độ lệch chuẩn kênh S**: Độ biến thiên của độ bão hòa màu
6. **Độ lệch chuẩn kênh V**: Độ biến thiên của độ sáng
7. **Tỷ lệ sắc tố melanin**: Thước đo lượng melanin trong da
8. **Chỉ số ITA (Individual Typology Angle)**: Góc typology cá nhân, dùng để phân loại màu da
9. **Giá trị trung bình kênh L (Lightness)**: Độ sáng trung bình trong không gian màu L*a*b*
10. **Giá trị trung bình kênh a**: Thành phần màu đỏ-xanh lá trong không gian màu L*a*b*
11. **Giá trị trung bình kênh b**: Thành phần màu vàng-xanh dương trong không gian màu L*a*b*
12. **Độ đồng nhất màu da**: Đo lường mức độ đồng đều của màu da
13. **Độ phản quang của da**: Đo lường khả năng phản chiếu ánh sáng của da
14. **Chỉ số đỏ của má**: Đo màu đỏ ở vùng má
15. **Chênh lệch màu giữa trán và cằm**: Chênh lệch màu sắc giữa vùng trán và cằm
16. **Hệ số tin cậy của phân loại màu da**: Mức độ tin cậy của phân loại màu da

### Vector Đặc Trưng Cảm Xúc (16 chiều)

1. **Khoảng cách giữa hai đuôi mắt và khóe miệng**: Đo sự căng/thả lỏng của cơ mặt
2. **Độ cong của miệng**: Đặc trưng cho nụ cười hoặc nét mặt buồn
3. **Độ mở của miệng**: Đo độ mở giữa môi trên và môi dưới
4. **Độ nhướng của lông mày**: Đo độ nâng lên của lông mày (biểu hiện ngạc nhiên/sợ hãi)
5. **Độ nhíu của lông mày**: Đo độ cau mày (biểu hiện tức giận)
6. **Độ mở của mắt**: Đo độ mở của mí mắt
7. **Chỉ số co cơ mặt AU4**: Đo hoạt động của cơ nhíu lông mày
8. **Chỉ số co cơ mặt AU12**: Đo hoạt động của cơ kéo khóe miệng (cười)
9. **Chỉ số co cơ mặt AU15**: Đo hoạt động của cơ kéo khóe miệng xuống (buồn)
10. **Chỉ số co cơ mặt AU1**: Đo hoạt động của cơ nâng phần trong lông mày
11. **Chỉ số co cơ mặt AU2**: Đo hoạt động của cơ nâng phần ngoài lông mày
12. **Chỉ số co cơ mặt AU6**: Đo hoạt động của cơ mắt (vui vẻ thật)
13. **Chỉ số co cơ mặt AU7**: Đo hoạt động của cơ xung quanh mắt (nheo mắt)
14. **Chỉ số co cơ mặt AU9**: Đo hoạt động của cơ nhăn mũi (ghê tởm)
15. **Chỉ số đồng nhất cảm xúc**: Mức độ phát hiện cùng một cảm xúc từ các đặc trưng khác nhau
16. **Hệ số tin cậy của cảm xúc**: Mức độ tin cậy trong dự đoán cảm xúc

## Giấy Phép
