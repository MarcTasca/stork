from setuptools import setup, find_packages

setup(
    name="stork",
    version="0.1",
    description="Surrogate gradient library based on pytorch",
    author="Friedemann Zenke",
    author_email="fzenke@gmail.com",
    license="MIT",
    packages=find_packages(),
    package_data={"stork.periodic_reset": ["README.md", "assets/*.png"]},
    zip_safe=False,
)
