import setuptools

setuptools.setup(
    name="aimless",
    version="0.0.1",
    author="Artificial Intelligence and Music League for Effective Source Separation",
    author_email="chin-yun.yu@qmul.ac.uk",
    packages=setuptools.find_packages(exclude=["tests", "tests.*", "data", "data.*"]),
    install_requires=["torch", "pytorch-lightning", "torch_fftconv"],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
