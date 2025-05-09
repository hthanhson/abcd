"""
Cấu hình kết nối MySQL cho ứng dụng
"""

# Thông tin kết nối MySQL - cần điều chỉnh theo môi trường thực tế
MYSQL_CONFIG = {
    'host': '192.168.147.1',
    'user': 'webuser',        # Thay đổi thành username của bạn
    'password': '123!',        # Thay đổi thành password của bạn
    'database': 'hcsdl',
    'port': 3306,
    'raise_on_warnings': True,
    'use_pure': True
} 