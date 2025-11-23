"""
Setup script for H3 Optimizer package.
"""

from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='h3_optimizer',
    version='0.2.0',
    author='Nuno Cardoso',
    author_email='nuno@example.com',
    description='Thermodynamically efficient optimizer with Lipschitz-adaptive learning rates and information-weighted sampling',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/nfocardoso/EMSTI',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'License :: OSI Approved :: MIT License',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: POSIX :: Linux',
    ],
    python_requires='>=3.7',
    install_requires=[
        'torch>=1.9.0',
        'numpy>=1.19.0',
        'torchvision>=0.10.0',
        'pandas>=1.3.0',
        'matplotlib>=3.3.0',
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
            'pynvml>=11.0.0',
        ],
        'notebooks': [
            'jupyter>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'h3-report=h3.cli:main',
        ],
    },
    keywords='optimization thermodynamics energy-efficient machine-learning deep-learning green-ai pytorch h3 lipschitz-adaptive information-weighted-sampling',
    project_urls={
        'Bug Reports': 'https://github.com/nfocardoso/EMSTI/issues',
        'Source': 'https://github.com/nfocardoso/EMSTI',
        'Documentation': 'https://github.com/nfocardoso/EMSTI/blob/main/README.md',
    },
    license='MIT',
)
