from setuptools import setup, find_packages

setup(
    name="bomax",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "numpy",
        "pandas",
        "matplotlib",
        "torch",
        "gpytorch",
        "botorch",
        "scipy",
        "scikit-learn",
    ],
    python_requires=">=3.8",
)
