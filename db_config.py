# Database Configuration

# Cấu hình kết nối cơ sở dữ liệu
# Chỉnh sửa các thông số này phù hợp với môi trường của bạn

# Database host - địa chỉ máy chủ MySQL
# Thay đổi nếu cơ sở dữ liệu không chạy trên host localhost
DB_HOST = '192.168.147.1'  # Thay đổi thành localhost nếu MySQL chạy trên cùng máy

# Database user - tên người dùng MySQL
DB_USER = 'webuser'

# Database password - mật khẩu MySQL
DB_PASSWORD = '123!'

# Database name - tên cơ sở dữ liệu
DB_NAME = 'hcsdl'

# Database port (default MySQL port is 3306)
DB_PORT = 3306

# Timeout cho kết nối MySQL (giây)
DB_TIMEOUT = 5

# Cấu hình thử lại kết nối
DB_MAX_RETRIES = 3  # Số lần thử lại tối đa
DB_RETRY_DELAY = 1  # Thời gian chờ giữa các lần thử (giây)

# Kiểm tra cấu hình khi import module
try:
    import mysql.connector
    print("Checking database connection...")
    conn = mysql.connector.connect(
        host=DB_HOST,
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
        port=DB_PORT,
        connection_timeout=DB_TIMEOUT
    )
    conn.close()
    print("Database connection successful!")
except Exception as e:
    print(f"Warning: Could not connect to database with current configuration: {e}")
    print("You may need to update the db_config.py file with correct connection details.")
    print("Current settings:")
    print(f"  Host: {DB_HOST}")
    print(f"  User: {DB_USER}")
    print(f"  Database: {DB_NAME}")
    print(f"  Port: {DB_PORT}")
