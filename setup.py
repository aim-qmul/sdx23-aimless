import setuptools

setuptools.setup(
    name="aimless",
    version="0.0.1",
    author="AIM",
    author_email="chin-yun.yu@qmul.ac.uk",
    packages=setuptools.find_packages(),
    install_requires=['torch'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
