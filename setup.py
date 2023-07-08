from pathlib import Path

from setuptools import find_packages, setup

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="csgoml",
    version="0.0.1",
    packages=find_packages(),
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "awpy>=1.2.3",
        "boto3~=1.28.1",
        "imageio~=2.28.0",
        "matplotlib~=3.7.0",
        "numba~=0.57.1",
        "numpy>=1.20.0",
        "patool~=1.12",
        "polars[numpy]~=0.18.6",
        "PyMySQL[rsa]~=1.1.0",
        "requests~=2.31.0",
        "requests_ip_rotator~=1.0.14",
        "scikit-learn~=1.3.0",
        "scikit-learn-extra~=0.3.0",
        "scipy~=1.10.0",
        "sympy~=1.12.0",
        "tensorflow~=2.13.0",
        "tqdm>=4.65.0",
        "watchdog~=3.0.0",
    ],
    # metadata to display on PyPI
    author="Jan-Eric Nitschke",
    author_email="janericnitschke@gmail.com",
    description="Counter-Strike: Global Offensive analysis functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="esports sports-analytics csgo counter-strike",
    url="https://github.com/JanEricNitschke/csgoml",
    project_urls={
        "Issues": "https://github.com/JanEricNitschke/csgoml/issues",
        "GitHub": "https://github.com/JanEricNitschke/csgoml/",
    },
    classifiers=["License :: OSI Approved :: MIT License"],
)
