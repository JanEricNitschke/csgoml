from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="csgoml",
    version="0.0.1",
    packages=find_packages(),
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "awpy>=1.2.1",
        "boto3>=1.20.48",
        "pymysql[rsa]>=1.0.2",
        "patool>=1.12",
        "watchdog>=2.1.6",
        "requests_ip_rotator>=1.0.10",
        "numpy>=1.18.1",
        "matplotlib>=3.1.2",
        "sympy>=1.10.1",
        "numba>=0.56.2",
        "imageio>=2.9.0",
        "requests>=2.25.1",
        "tqdm>=4.55.2",
        "pandas>=1.3.5",
        "scikit-learn>=1.0.2",
        "scikit-learn-extra>=0.2.0",
        "tensorflow>=2.8.0rc1",
        "scipy>=1.7.3",
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
