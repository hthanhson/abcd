from setuptools import setup, find_packages

setup(
    name="face_recognition_system",
    version="0.1.0",
    description="A system for storing and searching human faces",
    author="Your Name",
    author_email="your.email@example.com",
    packages=find_packages(),
    install_requires=[
        "Flask==2.0.1",
        "face-recognition==1.3.0",
        "opencv-python==4.5.3.56",
        "numpy==1.21.2",
        "Pillow==8.3.2",
        "scikit-learn==0.24.2",
        "dlib==19.22.1",
        "deepface==0.0.75",
        "Werkzeug==2.0.1",
        "matplotlib==3.4.3"
    ],
    python_requires=">=3.7",
) 