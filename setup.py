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
        "awpy @ git+https://github.com/JanEricNitschke/csgo@update-ci#egg=awpy",
        "boto3~=1.34.12",
        "imageio~=2.33.1",
        "matplotlib~=3.8.2",
        "numba~=0.59.0rc1",
        "numpy>=1.26,<=1.27",
        "patool~=2.0",
        "polars[numpy]~=0.20.3",
        "PyMySQL[rsa]~=1.1.0",
        "requests~=2.31.0",
        "requests_ip_rotator~=1.0.14",
        "scikit-learn~=1.3.2",
        "scikit-learn-extra~=0.3.0",
        "scipy~=1.11.4",
        "sympy==1.12",
        "tensorflow~=2.15.0",
        "tqdm~=4.66.1",
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
