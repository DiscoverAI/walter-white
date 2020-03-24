import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="walter_white",
    version="0.0.1",
    author="Daniel Schruhl",
    author_email="danielschruhl@gmail.com",
    description="A generative neural network for creating drugs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/discoverai/walter-white",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
