import os
import cv2
import numpy as np

def translate_gender(gender, to_vietnamese=True):
    """
    Translate gender between English and Vietnamese
    
    Args:
        gender: Gender string to translate
        to_vietnamese: Whether to translate to Vietnamese (True) or English (False)
        
    Returns:
        str: Translated gender string
    """
    if to_vietnamese:
        # English to Vietnamese
        translations = {
            'Man': 'Nam',
            'Woman': 'Nữ',
            'Unknown': 'Không xác định'
        }
        return translations.get(gender, 'Không xác định')
    else:
        # Vietnamese to English
        translations = {
            'Nam': 'Man',
            'Nữ': 'Woman',
            'Không xác định': 'Unknown'
        }
        return translations.get(gender, 'Unknown')

def translate_emotion(emotion, to_vietnamese=True):
    """
    Translate emotion between English and Vietnamese
    
    Args:
        emotion: Emotion string to translate
        to_vietnamese: Whether to translate to Vietnamese (True) or English (False)
        
    Returns:
        str: Translated emotion string
    """
    if to_vietnamese:
        # English to Vietnamese
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
    else:
        # Vietnamese to English
        translations = {
            'Tức giận': 'Angry',
            'Ghê tởm': 'Disgusted',
            'Sợ hãi': 'Fearful',
            'Vui vẻ': 'Happy',
            'Buồn bã': 'Sad',
            'Ngạc nhiên': 'Surprised',
            'Trung tính': 'Neutral',
            'Không xác định': 'Unknown'
        }
        return translations.get(emotion, 'Unknown')

def translate_skin_color(skin_color, to_vietnamese=True):
    """
    Translate skin color between English and Vietnamese
    
    Args:
        skin_color: Skin color string to translate
        to_vietnamese: Whether to translate to Vietnamese (True) or English (False)
        
    Returns:
        str: Translated skin color string
    """
    if to_vietnamese:
        # English to Vietnamese
        translations = {
            'White': 'Trắng',
            'Black': 'Đen',
            'Yellow': 'Vàng',
            'Unknown': 'Không xác định'
        }
        return translations.get(skin_color, 'Không xác định')
    else:
        # Vietnamese to English
        translations = {
            'Trắng': 'White',
            'Đen': 'Black',
            'Vàng': 'Yellow',
            'Không xác định': 'Unknown'
        }
        return translations.get(skin_color, 'Unknown')

def format_similarity(similarity, decimal_places=2):
    """
    Format similarity score as percentage string
    
    Args:
        similarity: Similarity score (0-1)
        decimal_places: Number of decimal places to include
        
    Returns:
        str: Formatted similarity percentage string
    """
    return f"{similarity * 100:.{decimal_places}f}%"

def validate_image_file(filename):
    """
    Validate if a file is an allowed image type
    
    Args:
        filename: Name of file to validate
        
    Returns:
        bool: True if file is an allowed image type, False otherwise
    """
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS 