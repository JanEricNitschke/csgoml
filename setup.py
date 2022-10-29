from setuptools import setup, find_packages

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="csgo_ml",
    version="0.0.1",
    packages=find_packages(),
    # Project uses reStructuredText, so ensure that the docutils get
    # installed or upgraded on the target machine
    install_requires=[
        "awpy>=1.1.9",
        "boto3",
        "pymysql",
        "patool",
        "watchdog",
        "requests_ip_rotator",
        "numpy>=1.18.1",
        "matplotlib>=3.1.2",
        "sympy>=1.10.1",
        "numba",
        "imageio>=2.9.0",
        "requests>=2.25.1",
        "tqdm>=4.55.2",
        "pandas>=0.25.3",
        "scikit-learn",
        "tensorflow",
    ],
    # metadata to display on PyPI
    author="Jan-Eric Nitschke",
    author_email="janericnitschke@gmail.com",
    description="Counter-Strike: Global Offensive analysis functions",
    long_description=long_description,
    long_description_content_type="text/markdown",
    keywords="esports sports-analytics csgo counter-strike",
    url="https://github.com/JanEricNitschke/csgo_ml",
    project_urls={
        "Issues": "https://github.com/JanEricNitschke/csgo_ml/issues",
        "GitHub": "https://github.com/JanEricNitschke/csgo_ml/",
    },
    classifiers=["License :: OSI Approved :: MIT License"],
)
