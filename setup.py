import setuptools

import gtFrame

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="gtFrame",
    author="bluePlatinum",
    version=gtFrame.__version__,
    author_email="jukic.rok@gmail.com",
    description="A simple library for working with frames of reference",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/bluePlatinum/gt-frame",
    package_dir={"": "gtFrame"},
    packages=setuptools.find_packages(where="gtFrame"),
    python_requires=">3.7"
)
