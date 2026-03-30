from setuptools import setup, find_packages

setup(
    name='transform_emr',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "pandas",
        "numpy",
        "scikit-learn",
        "tqdm",
        "matplotlib",
        "openpyxl"
    ],
)