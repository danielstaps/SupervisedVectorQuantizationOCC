from setuptools import find_packages, setup

PROJECT_URL = "https://github.com/si-cim/one-class-classifier_pytorch"
DOWNLOAD_URL = "https://github.com/si-cim/one-class-classifier_pytorch.git"

with open("README.md", "r") as fh:
    long_description = fh.read()

INSTALL_REQUIRES = [
    "prototorch_models",
    "pandas",
    "matplotlib",
]

ALL = INSTALL_REQUIRES

setup(
    name="prototorch_oneclass",
    version="0.0.1",
    author="Daniel Staps",
    author_email="staps@hs-mittweida.de",
    description="Implementation of One Class Classifier",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=PROJECT_URL,
    download_url=DOWNLOAD_URL,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
    packages=find_packages(),
    classifiers=[
        "Environment :: Console",
        "Natural Language :: English",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    zip_safe=False,
)
