import os
import cv2
import numpy as np

def translate_gender(gender):
    """
    Chuyển đổi giới tính từ tiếng Anh sang tiếng Việt
    
    Args:
        gender: Chuỗi giới tính cần chuyển đổi
        
    Returns:
        str: Chuỗi giới tính đã được chuyển sang tiếng Việt
    """
    # Bảng chuyển đổi từ tiếng Anh sang tiếng Việt
    translations = {
        'Man': 'Nam',
        'Woman': 'Nữ',
        'Unknown': 'Không xác định'
    }
    return translations.get(gender, 'Không xác định')

def translate_emotion(emotion):
    """
    Chuyển đổi cảm xúc từ tiếng Anh sang tiếng Việt
    
    Args:
        emotion: Chuỗi cảm xúc cần chuyển đổi
        
    Returns:
        str: Chuỗi cảm xúc đã được chuyển sang tiếng Việt
    """
    # Bảng chuyển đổi từ tiếng Anh sang tiếng Việt
    translations = {
        'Angry': 'Tức giận',
        'Disgusted': 'Ghê tởm',
        'Fearful': 'Sợ hãi',
        'Happy': 'Vui vẻ',
        'Sad': 'Buồn bã',
        'Surprised': 'Ngạc nhiên',
        'Neutral': 'Trung tính',
        'Unknown': 'Không xác định'
    }
    return translations.get(emotion, 'Không xác định')

def translate_skin_color(skin_color):
    """
    Chuyển đổi màu da từ tiếng Anh sang tiếng Việt
    
    Args:
        skin_color: Chuỗi màu da cần chuyển đổi
        
    Returns:
        str: Chuỗi màu da đã được chuyển sang tiếng Việt
    """
    # Bảng chuyển đổi từ tiếng Anh sang tiếng Việt
    translations = {
        'White': 'Trắng',
        'Black': 'Đen',
        'Yellow': 'Vàng',
        'Unknown': 'Không xác định'
    }
    return translations.get(skin_color, 'Không xác định') 