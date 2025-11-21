"""
Setup script for H3 Optimizer package.
"""

from setuptools import setup, find_packages
import os


# Read the contents of README file
def read_long_description():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, encoding='utf-8') as f:
            return f.read()
    return ""


setup(
    name='h3-optimizer',
    version='0.1.0',
    author='H3 Development Team',
    author_email='',
    description='Hierarchical Hessian-informed optimizer for efficient deep learning',
    long_description=read_long_description(),
    long_description_content_type='text/markdown',
    url='https://github.com/yourusername/h3-optimizer',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'torchvision>=0.10.0',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0',
            'pytest-cov>=2.12',
            'black>=21.0',
            'flake8>=3.9',
            'mypy>=0.910',
        ],
        'gpu': [
            'pynvml>=11.0.0',  # For GPU energy monitoring
        ],
        'notebooks': [
            'jupyter>=1.0.0',
            'matplotlib>=3.3.0',
            'pandas>=1.3.0',
        ],
    },
    keywords='deep-learning optimization pytorch machine-learning',
    project_urls={
        'Bug Reports': 'https://github.com/yourusername/h3-optimizer/issues',
        'Source': 'https://github.com/yourusername/h3-optimizer',
    },
)
