from setuptools import setup, find_packages

setup(
    name='ocr-data-toolkit',
    version='0.1.2',
    author='Muhammad Nouman Ahsan',
    author_email='naumanhsa965@gmail.com',
    description='A toolkit for generating synthetic data for OCR',
    long_description=open('README_PYPI.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/NaumanHSA/ocr-data-toolkit',
    license='MIT',
    packages=find_packages(),
    install_requires=[
        'Pillow>=11.2.1',
        'numpy>=1.24.1',
        'opencv-python>=4.2.0',
        'matplotlib>=3.5.1',
        'atpbar>=2.0.3',
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],    
    python_requires='>=3.7',
    include_package_data=True,
    zip_safe=False
)
