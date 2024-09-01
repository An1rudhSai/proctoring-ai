from setuptools import setup, find_packages

setup(
    name='proctoring_ai',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'opencv-python',
        'dlib',
        'numpy',
        'tensorflow',
        'wget',
        'flask'
    ],
    entry_points={
        'console_scripts': [
            'proctoring_ai=proctoring_ai.app:main',
        ],
    },
)
