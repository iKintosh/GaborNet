import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="GaborNet",
    version="0.1.0",
    author="Andrey Alekseev",
    author_email="ilekseev@gmail.com",
    description="Gabor layer implementation",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/iKintosh/GaborNet",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    tests_require=[
        "pytest",
        "pylint",
        "pycodestyle",
        "pep257",
    ],
    setup_requires=[
        "pytest-runner",
        "pytest-pylint",
        "pytest-pycodestyle",
        "pytest-pep257",
        "pytest-cov",
    ],
    install_requires=["numpy", "pytest==5.2.1", "torch"],
)
