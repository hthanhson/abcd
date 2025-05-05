import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def create_architecture_diagram():
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(5, 6.7, 'Hệ thống Nhận diện và Tìm kiếm Khuôn mặt - Kiến trúc', 
            fontsize=16, ha='center', weight='bold')
    
    # Create components
    components = [
        {'name': 'Người dùng', 'x': 5, 'y': 6, 'width': 2, 'height': 0.6, 'color': '#F8CBA6'},
        {'name': 'Giao diện Web\n(HTML/CSS/JS)', 'x': 5, 'y': 5, 'width': 2, 'height': 0.7, 'color': '#A6E1F8'},
        {'name': 'Flask Backend', 'x': 5, 'y': 4, 'width': 2, 'height': 0.6, 'color': '#A6F8C4'},
        
        {'name': 'Trích xuất\nđặc trưng', 'x': 3, 'y': 3, 'width': 1.5, 'height': 0.7, 'color': '#F8A6E1'},
        {'name': 'So sánh\ntương đồng', 'x': 5, 'y': 3, 'width': 1.5, 'height': 0.7, 'color': '#F8A6E1'},
        {'name': 'Lọc theo\nđặc trưng', 'x': 7, 'y': 3, 'width': 1.5, 'height': 0.7, 'color': '#F8A6E1'},
        
        {'name': 'face_recognition', 'x': 2, 'y': 2, 'width': 1.5, 'height': 0.6, 'color': '#C4A6F8'},
        {'name': 'OpenCV', 'x': 4, 'y': 2, 'width': 1.5, 'height': 0.6, 'color': '#C4A6F8'},
        {'name': 'DeepFace', 'x': 6, 'y': 2, 'width': 1.5, 'height': 0.6, 'color': '#C4A6F8'},
        {'name': 'Scikit-learn', 'x': 8, 'y': 2, 'width': 1.5, 'height': 0.6, 'color': '#C4A6F8'},
        
        {'name': 'CSDL Đặc trưng\n(Pickle File)', 'x': 3, 'y': 1, 'width': 2, 'height': 0.7, 'color': '#F8D3A6'},
        {'name': 'Thư mục Ảnh\n(data_test)', 'x': 7, 'y': 1, 'width': 2, 'height': 0.7, 'color': '#F8D3A6'},
    ]
    
    # Draw components
    for comp in components:
        rect = patches.Rectangle((comp['x'] - comp['width']/2, comp['y'] - comp['height']/2), 
                                comp['width'], comp['height'], 
                                linewidth=1, edgecolor='black', facecolor=comp['color'], alpha=0.9)
        ax.add_patch(rect)
        ax.text(comp['x'], comp['y'], comp['name'], ha='center', va='center', fontsize=10)
    
    # Draw arrows
    arrows = [
        # User interaction
        {'start': (5, 5.7), 'end': (5, 5.35), 'label': 'Tương tác'},
        {'start': (5, 4.65), 'end': (5, 4.3), 'label': 'HTTP Request/Response'},
        
        # Backend to modules
        {'start': (4.3, 3.7), 'end': (3.75, 3.35), 'label': ''},
        {'start': (5, 3.7), 'end': (5, 3.35), 'label': ''},
        {'start': (5.7, 3.7), 'end': (6.25, 3.35), 'label': ''},
        
        # Modules to libraries
        {'start': (3, 2.65), 'end': (2.75, 2.3), 'label': ''},
        {'start': (3.5, 2.65), 'end': (4.25, 2.3), 'label': ''},
        {'start': (6.5, 2.65), 'end': (6.75, 2.3), 'label': ''},
        {'start': (7, 2.65), 'end': (7.25, 2.3), 'label': ''},
        
        # Libraries to data
        {'start': (3.25, 1.7), 'end': (3.5, 1.35), 'label': 'Đọc/Ghi'},
        {'start': (7.75, 1.7), 'end': (7.5, 1.35), 'label': 'Đọc'},
        
        # Cross connections
        {'start': (4, 2), 'end': (6.5, 1.35), 'label': ''},
        {'start': (6, 2), 'end': (4, 1.35), 'label': ''},
    ]
    
    for arrow in arrows:
        ax.annotate('', xy=arrow['end'], xytext=arrow['start'],
                   arrowprops=dict(arrowstyle='->', lw=1, color='black'))
        
        # Add label if exists
        if arrow['label']:
            mid_x = (arrow['start'][0] + arrow['end'][0]) / 2
            mid_y = (arrow['start'][1] + arrow['end'][1]) / 2
            ax.text(mid_x + 0.2, mid_y, arrow['label'], fontsize=8, ha='left', va='center')
    
    # Data flow explanation
    flow_steps = [
        "1. Người dùng tương tác với giao diện web",
        "2. Giao diện gửi yêu cầu tới Flask Backend",
        "3. Backend điều hướng yêu cầu tới các module xử lý",
        "4. Các module sử dụng thư viện để xử lý ảnh và nhận diện khuôn mặt",
        "5. Đặc trưng được lưu trữ hoặc so sánh với CSDL hiện có",
        "6. Kết quả trả về cho người dùng qua giao diện web"
    ]
    
    flow_text = "\n".join(flow_steps)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.2)
    ax.text(1.5, 0.5, flow_text, fontsize=9, bbox=props, va='center')
    
    # Save the diagram
    plt.tight_layout()
    plt.savefig('architecture.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Architecture diagram generated successfully!")

if __name__ == "__main__":
    create_architecture_diagram() 