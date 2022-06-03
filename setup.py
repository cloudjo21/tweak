from setuptools import setup, find_packages
import setuptools

with open("requirements.txt") as f:
    required = f.read().splitlines()

setup(
    name="tweak",
    version="0.0.3",
    url="https://bitbucket.org/peterleecodiit/tweak.git",
    packages=find_packages("src"),
    package_dir={"tweak": "src/tweak"},
    python_requires=">=3.6",
    long_description=open("README.md").read(),
    install_requires=required,
)

