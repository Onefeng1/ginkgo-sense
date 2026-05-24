# -*- coding: utf-8 -*-
"""
GinkgoSense 包配置
"""
from setuptools import setup, find_packages

setup(
    name='ginkgo-sense',
    version='1.0.0',
    author='郑俊锋',
    author_email='onefeng1@users.noreply.github.com',
    description='基于深度学习的白果智能识别系统',
    long_description=open('README.md', encoding='utf-8').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Onefeng1/ginkgo-sense',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'torch>=1.12',
        'torchvision>=0.13',
        'flask>=2.3',
        'pillow>=9.0',
        'numpy>=1.21',
        'werkzeug>=2.3',
    ],
    extras_require={
        'dev': [
            'pytest>=7.0',
            'pytest-cov>=4.0',
            'flake8>=6.0',
            'black>=23.0',
            'isort>=5.12',
        ],
        'export': [
            'onnx>=1.14',
            'onnxruntime>=1.15',
        ],
        'visualization': [
            'matplotlib>=3.5',
            'scikit-learn>=1.2',
        ],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Image Recognition',
    ],
    entry_points={
        'console_scripts': [
            'ginkgo-train=model.train:main',
            'ginkgo-eval=scripts.evaluate:main',
            'ginkgo-export=scripts.export_model:main',
        ],
    },
)
