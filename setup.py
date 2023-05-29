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
        "boto3>=1.20.48",
        "PyMySQL[rsa]>=1.0.2",
        "patool>=1.12",
        "watchdog>=2.1.6",
        "requests_ip_rotator>=1.0.10",
        "numpy>=1.20.0",
        "matplotlib>=3.7.1",
        "sympy>=1.12.0",
        "numba>=0.57.0",
        "imageio>=2.28.1",
        "requests>=2.30.0",
        "tqdm>=4.65.0",
        "polars[numpy]>=0.17.15",
        "scikit-learn>=1.2.2",
        "scikit-learn-extra>=0.3.0",
        "tensorflow>=2.12.0",
        "scipy>=1.10.1",
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
