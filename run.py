#!/usr/bin/env python3
"""
Main script for running the Face Recognition System
This script provides a menu to run different parts of the system.
"""

import os
import sys
import subprocess
import time

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def print_header():
    """Print header for the application"""
    print("=" * 60)
    print("          HỆ THỐNG NHẬN DIỆN VÀ TÌM KIẾM MẶT NGƯỜI")
    print("=" * 60)
    print("Chức năng:")
    print("  1. Xây dựng cơ sở dữ liệu đặc trưng khuôn mặt")
    print("  2. Khởi động ứng dụng web")
    print("  3. Chạy demo nhận diện khuôn mặt")
    print("  4. Chạy benchmark hiệu suất hệ thống")
    print("  5. Tạo sơ đồ kiến trúc hệ thống")
    print("  0. Thoát")
    print("-" * 60)

def run_option(option):
    """Run the selected option"""
    try:
        if option == 1:
            # Build database
            print("Đang xây dựng cơ sở dữ liệu đặc trưng khuôn mặt...")
            subprocess.run([sys.executable, 'app.py', 'build_db'], check=True)
            print("\nĐã xây dựng cơ sở dữ liệu thành công!")
            
        elif option == 2:
            # Run web app
            print("Đang khởi động ứng dụng web...")
            print("Truy cập http://localhost:5000 để sử dụng ứng dụng")
            print("Ấn Ctrl+C để dừng máy chủ")
            subprocess.run([sys.executable, 'app.py'], check=True)
            
        elif option == 3:
            # Run demo
            print("Đang chạy demo nhận diện khuôn mặt...")
            subprocess.run([sys.executable, 'demo.py'], check=True)
            print("\nDemo đã hoàn tất. Kiểm tra thư mục demo_output để xem kết quả.")
            
        elif option == 4:
            # Run benchmark
            print("Đang chạy benchmark hiệu suất hệ thống...")
            subprocess.run([sys.executable, 'benchmark.py'], check=True)
            print("\nBenchmark đã hoàn tất. Kiểm tra thư mục benchmark_results để xem kết quả.")
            
        elif option == 5:
            # Generate architecture diagram
            print("Đang tạo sơ đồ kiến trúc hệ thống...")
            subprocess.run([sys.executable, 'generate_architecture.py'], check=True)
            print("\nĐã tạo sơ đồ kiến trúc thành công! Xem file architecture.png")
            
        else:
            print("Lựa chọn không hợp lệ.")
            
    except subprocess.CalledProcessError as e:
        print(f"\nLỗi khi thực thi: {e}")
    except FileNotFoundError as e:
        print(f"\nLỗi: Không tìm thấy file - {e}")
    except KeyboardInterrupt:
        print("\nĐã hủy thao tác.")
    
    input("\nNhấn Enter để tiếp tục...")

def check_dependencies():
    """Check if all the required dependencies are installed"""
    required_files = ['app.py', 'demo.py', 'benchmark.py', 'generate_architecture.py', 'requirements.txt']
    
    for file in required_files:
        if not os.path.exists(file):
            print(f"Lỗi: Không tìm thấy file {file}")
            print("Vui lòng đảm bảo bạn đã tải đầy đủ các file của dự án.")
            return False
    
    print("Kiểm tra các thư viện phụ thuộc...")
    try:
        # Try importing some key libraries
        import flask
        import face_recognition
        import cv2
        import numpy
        import matplotlib
        
        print("Tất cả các thư viện phụ thuộc đã được cài đặt đúng.")
        return True
        
    except ImportError as e:
        print(f"Lỗi: Thiếu thư viện - {e}")
        print("\nVui lòng cài đặt các thư viện phụ thuộc:")
        print("pip install -r requirements.txt")
        
        choice = input("Bạn có muốn cài đặt ngay bây giờ? (y/n): ")
        if choice.lower() == 'y':
            try:
                subprocess.run(['pip', 'install', '-r', 'requirements.txt'], check=True)
                print("Đã cài đặt các thư viện phụ thuộc thành công!")
                return True
            except subprocess.CalledProcessError:
                print("Lỗi khi cài đặt thư viện phụ thuộc.")
                return False
        return False

def main():
    """Main function"""
    if not check_dependencies():
        input("Nhấn Enter để thoát...")
        sys.exit(1)
    
    while True:
        clear_screen()
        print_header()
        
        try:
            choice = int(input("Chọn chức năng (0-5): "))
            
            if choice == 0:
                print("Cảm ơn bạn đã sử dụng hệ thống!")
                break
                
            if 1 <= choice <= 5:
                clear_screen()
                run_option(choice)
            else:
                print("Lựa chọn không hợp lệ. Vui lòng chọn từ 0-5.")
                time.sleep(1)
                
        except ValueError:
            print("Vui lòng nhập một số từ 0-5.")
            time.sleep(1)
        except KeyboardInterrupt:
            print("\nĐang thoát chương trình...")
            break

if __name__ == "__main__":
    main() 