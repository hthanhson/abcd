# Hệ Thống Nhận Dạng Khuôn Mặt Tiên Tiến

## Giới Thiệu

Dự án này là một hệ thống nhận dạng khuôn mặt tiên tiến sử dụng trí tuệ nhân tạo để phân tích và trích xuất các đặc trưng chi tiết từ hình ảnh khuôn mặt. Hệ thống cung cấp khả năng phân tích đa chiều bao gồm giới tính, màu da, cảm xúc và mã hóa khuôn mặt.

## Tính Năng Chính

- **Trích Xuất Đặc Trưng Khuôn Mặt**: 
  - Vector đặc trưng 173 chiều
  - Bao gồm mã hóa khuôn mặt 128 chiều
  - Vector giới tính 15 chiều
  - Vector màu da 15 chiều
  - Vector cảm xúc 15 chiều

- **Phân Tích Chi Tiết**:
  - Nhận dạng giới tính (Nam/Nữ)
  - Phân loại màu da (Trắng/Đen/Vàng)
  - Nhận dạng cảm xúc (Vui/Buồn/Giận/Ngạc Nhiên/Sợ Hãi/Ghê Tởm/Trung Tính)

- **Tìm Kiếm Khuôn Mặt Tương Tự**:
  - Sử dụng độ tương đồng cosine
  - Hỗ trợ lọc theo nhiều tiêu chí

## Yêu Cầu Hệ Thống

### Phần Mềm
- Python 3.8+
- OpenCV
- NumPy
- MySQL Connector
- Flask
- SciPy

### Cấu Hình Phần Cứng
- RAM: 8GB trở lên
- CPU: Hỗ trợ OpenCV và xử lý song song
- Không yêu cầu GPU (nhưng khuyến nghị có GPU để tăng tốc độ)

## Cấu Trúc Dự Án
```
project/
│
├── feature_extraction.py     # Trích xuất đặc trưng khuôn mặt
├── emotion_detection.py      # Nhận dạng cảm xúc
├── gender_detection.py       # Nhận dạng giới tính
├── skin_classification.py    # Phân loại màu da
├── database.py               # Quản lý cơ sở dữ liệu
├── app.py                    # Ứng dụng web Flask
└── utils.py                  # Các hàm tiện ích
```

## Chi Tiết Kỹ Thuật

### Vector Đặc Trưng Chi Tiết

#### 1. Vector Mã Hóa Khuôn Mặt (128 chiều)
- **Mục Đích**: Tạo biểu diễn số duy nhất và ổn định của khuôn mặt
- **Chi Tiết Kỹ Thuật**:
  - Trích xuất từ 16 vùng khác nhau (4x4 grid)
  - Sử dụng gradient và histogram hướng
  - Mỗi chiều biểu thị một đặc trưng cụ thể của khuôn mặt
- **Các Giá Trị**:
  - Chiều 0-31: Đặc trưng vùng mắt trái
  - Chiều 32-63: Đặc trưng vùng mắt phải
  - Chiều 64-95: Đặc trưng vùng mũi và má
  - Chiều 96-127: Đặc trưng vùng miệng và cằm
- **Ứng Dụng**:
  - So sánh độ tương đồng khuôn mặt
  - Nhận dạng cá nhân
  - Tìm kiếm khuôn mặt trong cơ sở dữ liệu

#### 2. Vector Giới Tính (15 chiều)
- **Mục Đích**: Đặc trưng và phân loại giới tính của khuôn mặt
- **Chi Tiết Từng Chiều**:
  - **Chiều 0**: Giới tính dự đoán - Giá trị số hóa (0=Nam, 1=Nữ, 2=Không xác định)
  - **Chiều 1**: Độ tin cậy - Mức độ chắc chắn của dự đoán giới tính (0-1)
  - **Chiều 2**: One-hot cho Nữ - Giá trị 1 nếu là Nữ, 0 nếu không
  - **Chiều 3**: One-hot cho Nam - Giá trị 1 nếu là Nam, 0 nếu không
  - **Chiều 4**: Tỷ lệ khuôn mặt - Tỷ lệ chiều cao/chiều rộng (chuẩn hóa 0-1)
  - **Chiều 5**: Độ mịn da - Đo lường texture của da (chuẩn hóa 0-1)
  - **Chiều 6**: Đặc trưng hàm - Mức độ nổi bật của đường hàm (chuẩn hóa 0-1)
  - **Chiều 7**: Đặc trưng mắt - Đặc điểm vùng mắt (chuẩn hóa 0-1)
  - **Chiều 8**: One-hot encoding phân loại Nam (0-2)
  - **Chiều 9**: One-hot encoding phân loại Nữ (0-2)
  - **Chiều 10**: One-hot encoding phân loại Không xác định (0-2)
  - **Chiều 11**: Giới tính dự đoán (normalized) - Giá trị 0-1 biểu thị giới tính
  - **Chiều 12**: Độ tin cậy - Lặp lại độ tin cậy của dự đoán
  - **Chiều 13**: One-hot cho Nữ - Lặp lại giá trị cho Nữ
  - **Chiều 14**: One-hot cho Nam - Lặp lại giá trị cho Nam
- **Ứng Dụng**:
  - Phân loại giới tính tự động
  - Tìm kiếm theo giới tính
  - Phân tích nhân khẩu học

#### 3. Vector Cảm Xúc (15 chiều)
- **Mục Đích**: Đặc trưng và phân loại cảm xúc thể hiện trên khuôn mặt
- **Chi Tiết Từng Chiều**:
  - **Chiều 0**: Trung bình vùng mắt - Độ sáng trung bình vùng mắt (0-1)
  - **Chiều 1**: Độ lệch chuẩn vùng mắt - Texture và chi tiết vùng mắt (0-1)
  - **Chiều 2**: Trung bình vùng miệng - Độ sáng trung bình vùng miệng (0-1)
  - **Chiều 3**: Độ lệch chuẩn vùng miệng - Texture và chi tiết vùng miệng (0-1)
  - **Chiều 4**: Độ cong môi trên - Độ cong của môi trên (chuẩn hóa, giá trị âm=cong xuống)
  - **Chiều 5**: Độ cong môi dưới - Độ cong của môi dưới (chuẩn hóa, giá trị âm=cong lên)
  - **Chiều 6**: Gradient toàn cục - Mức độ thay đổi độ sáng toàn bộ khuôn mặt (0-1)
  - **Chiều 7**: Độ lệch chuẩn gradient - Đo lường sự phức tạp của khuôn mặt (0-1)
  - **Chiều 8**: Cường độ lông mày - Đo lường gradient và đặc điểm lông mày (0-1)
  - **Chiều 9**: One-hot encoding cho cảm xúc Vui vẻ (0-7)
  - **Chiều 10**: One-hot encoding cho cảm xúc Buồn (0-7)
  - **Chiều 11**: One-hot encoding cho cảm xúc Giận dữ (0-7)
  - **Chiều 12**: One-hot encoding cho cảm xúc Ngạc nhiên (0-7)
  - **Chiều 13**: One-hot encoding cho cảm xúc Sợ hãi (0-7)
  - **Chiều 14**: One-hot encoding cho cảm xúc Ghê tởm hoặc Trung tính (0-7)
- **Ứng Dụng**:
  - Nhận dạng biểu cảm khuôn mặt
  - Phân tích tâm lý từ hình ảnh
  - Tương tác người-máy thông minh

#### 4. Vector Màu Da (15 chiều)
- **Mục Đích**: Đặc trưng và phân loại màu da trên khuôn mặt
- **Chi Tiết Từng Chiều**:
  - **Chiều 0**: Trung bình kênh đỏ (R) - Giá trị trung bình kênh đỏ trên vùng da (0-1)
  - **Chiều 1**: Trung bình kênh xanh lá (G) - Giá trị trung bình kênh xanh lá trên vùng da (0-1)
  - **Chiều 2**: Trung bình kênh xanh dương (B) - Giá trị trung bình kênh xanh dương trên vùng da (0-1)
  - **Chiều 3**: Trung bình kênh màu (H) - Trung bình tông màu trong không gian HSV (0-1)
  - **Chiều 4**: Trung bình độ bão hòa (S) - Trung bình độ bão hòa màu trong không gian HSV (0-1)
  - **Chiều 5**: Trung bình độ sáng (V) - Trung bình độ sáng trong không gian HSV (0-1)
  - **Chiều 6**: Độ tin cậy - Mức độ tin cậy của phân loại màu da (0-1)
  - **Chiều 7**: Độ sáng tổng thể - Tính theo công thức từ RGB (0-1)
  - **Chiều 8**: Tỷ lệ R/B - Tỷ lệ giữa kênh đỏ và kênh xanh dương (>1 cho da ấm)
  - **Chiều 9**: Tỷ lệ R/G - Tỷ lệ giữa kênh đỏ và kênh xanh lá (biểu thị đặc điểm da)
  - **Chiều 10**: Độ sáng (L) - Giá trị độ sáng trong không gian LAB (0-1)
  - **Chiều 11**: Giá trị (b) - Trục vàng-xanh trong không gian LAB (cao=vàng, thấp=xanh)
  - **Chiều 12**: One-hot encoding cho da Trắng (1=Trắng, 0=Khác)
  - **Chiều 13**: One-hot encoding cho da Đen (1=Đen, 0=Khác)
  - **Chiều 14**: One-hot encoding cho da Vàng (1=Vàng, 0=Khác)
- **Ứng Dụng**:
  - Phân loại màu da
  - Điều chỉnh tông màu tự động
  - Nghiên cứu nhân chủng học

#### 5. Vector Tổng Hợp (173 chiều)
- **Mục Đích**: Tạo biểu diễn toàn diện về khuôn mặt bao gồm tất cả các đặc trưng
- **Chi Tiết Kỹ Thuật**:
  - Kết hợp 4 vector thành 1 vector duy nhất
  - Bao gồm: Vector Mã hóa (128) + Vector Giới tính (15) + Vector Cảm xúc (15) + Vector Màu da (15)
- **Ứng Dụng**:
  - Tìm kiếm khuôn mặt tương tự toàn diện
  - Phân tích đa chiều
  - Nghiên cứu về biểu hiện khuôn mặt

## Chi Tiết Hàm Chức Năng

### Trích Xuất Đặc Trưng (`feature_extraction.py`)

#### `extract_face(image)`
- **Mục Đích**: Trích xuất vùng khuôn mặt từ ảnh đầu vào
- **Kỹ Thuật**:
  - Sử dụng Haar Cascade Classifier để phát hiện khuôn mặt
  - Hỗ trợ nhiều tỷ lệ và kích thước khuôn mặt
  - Thêm margin để bao quanh toàn bộ khuôn mặt
- **Đầu Ra**: Ảnh khuôn mặt được cắt và điều chỉnh kích thước

#### `create_face_encoding(face_image)`
- **Mục Đích**: Tạo vector mã hóa 128 chiều cho khuôn mặt
- **Kỹ Thuật**:
  - Chia khuôn mặt thành 16 vùng (4x4 grid)
  - Tính toán gradient và histogram hướng cho mỗi vùng
  - Chuẩn hóa vector để đảm bảo tính ổn định
- **Đầu Ra**: Vector 128 chiều biểu diễn đặc trưng khuôn mặt

#### `extract_features(image_path/image_array)`
- **Mục Đích**: Trích xuất toàn bộ đặc trưng từ ảnh khuôn mặt
- **Các Bước**:
  1. Trích xuất khuôn mặt
  2. Tạo mã hóa khuôn mặt
  3. Nhận dạng giới tính
  4. Phân loại màu da
  5. Nhận dạng cảm xúc
- **Đầu Ra**: Từ điển chứa các vector và nhãn chi tiết

### Nhận Dạng Cảm Xúc (`emotion_detection.py`)

#### `detect_emotion(face_image)`
- **Mục Đích**: Nhận dạng cảm xúc từ ảnh khuôn mặt
- **Kỹ Thuật Phân Tích**:
  - Phân tích vùng miệng, mắt, lông mày
  - Sử dụng gradient, độ lệch chuẩn, và ngưỡng
  - Tính điểm cho 7 loại cảm xúc
- **Đầu Ra**: Loại cảm xúc và độ tin cậy

#### `get_emotion_vector(face_image)`
- **Mục Đích**: Tạo vector đặc trưng cảm xúc 15 chiều
- **Các Đặc Trưng**:
  - Gradient toàn cục
  - Đặc điểm vùng miệng và mắt
  - Mã hóa one-hot cho loại cảm xúc
  - Độ tin cậy và cường độ cảm xúc
- **Đầu Ra**: Vector 15 chiều mô tả cảm xúc

### Nhận Dạng Giới Tính (`gender_detection.py`)

#### `detect_gender(face_image)`
- **Mục Đích**: Nhận dạng giới tính từ ảnh khuôn mặt
- **Kỹ Thuật Phân Tích**:
  - Phân tích texture da
  - Đánh giá vùng má, mắt, hàm
  - Sử dụng face encoding để tăng độ chính xác
- **Đầu Ra**: Giới tính (Nam/Nữ) và độ tin cậy

#### `get_gender_vector(face_image)`
- **Mục Đích**: Tạo vector đặc trưng giới tính 15 chiều
- **Các Đặc Trưng**:
  - Gradient toàn cục
  - Đặc điểm vùng hàm và mắt
  - Mã hóa one-hot cho giới tính
  - Độ tin cậy và cường độ giới tính
- **Đầu Ra**: Vector 15 chiều mô tả giới tính

### Phân Loại Màu Da (`skin_classification.py`)

#### `classify_skin_color(face_image)`
- **Mục Đích**: Phân loại màu da từ ảnh khuôn mặt
- **Kỹ Thuật Phân Tích**:
  - Sử dụng nhiều không gian màu (HSV, YCrCb, LAB)
  - Phân tích độ sáng, tỷ lệ màu
  - Tính điểm cho 3 loại màu da (Trắng/Đen/Vàng)
- **Đầu Ra**: Loại màu da và độ tin cậy

#### `get_skin_vector(face_image)`
- **Mục Đích**: Tạo vector đặc trưng màu da 15 chiều
- **Các Đặc Trưng**:
  - Gradient toàn cục
  - Đặc điểm vùng da và má
  - Mã hóa one-hot cho màu da
  - Độ tin cậy và cường độ màu da
- **Đầu Ra**: Vector 15 chiều mô tả màu da

### Quản Lý Cơ Sở Dữ Liệu (`database.py`)

#### `add_image_to_database(image_path, features)`
- **Mục Đích**: Thêm ảnh và các đặc trưng vào cơ sở dữ liệu
- **Các Bước**:
  1. Kết nối đến cơ sở dữ liệu
  2. Chèn thông tin ảnh
  3. Lưu trữ các vector đặc trưng
- **Đầu Ra**: Trạng thái thành công/thất bại

#### `find_similar_faces(query_features)`
- **Mục Đích**: Tìm kiếm khuôn mặt tương tự trong cơ sở dữ liệu
- **Kỹ Thuật**:
  - Sử dụng độ tương đồng cosine
  - Hỗ trợ lọc theo giới tính, màu da, cảm xúc
- **Đầu Ra**: Danh sách khuôn mặt tương tự nhất

#### `build_database(folder_path)`
- **Mục Đích**: Xây dựng cơ sở dữ liệu từ thư mục ảnh
- **Các Bước**:
  1. Quét tìm các tệp ảnh
  2. Trích xuất đặc trưng từng ảnh
  3. Lưu trữ vào cơ sở dữ liệu
- **Đầu Ra**: Số lượng ảnh được xử lý thành công

