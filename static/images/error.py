from PIL import Image, ImageDraw, ImageFont
import os

# Tạo một ảnh trống với nền xám nhạt
img = Image.new('RGB', (300, 200), color=(245, 245, 245))
draw = ImageDraw.Draw(img)

# Vẽ khung viền
draw.rectangle([(0, 0), (299, 199)], outline=(200, 200, 200))

# Thêm text
draw.text((80, 80), "Không tải được ảnh", fill=(100, 100, 100))
draw.text((120, 100), "Error", fill=(255, 0, 0))

# Lưu ảnh
img.save('error.png')
print("Đã tạo ảnh lỗi: error.png") 