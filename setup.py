from setuptools import find_packages, setup
setup(
    name="bandits",
    packages=find_packages(include=["bandits"]),
    version="0.0.1",
    description="Toolkit for bandits strategies",
    author="Me",
    license="MIT",
    python_requires=">=3.7",
    install_requires=[
        "numpy==1.19.1",
        "matplotlib==3.3.1"
        ],
    setup_requires=["pytest-runner"],
    tests_require=["pytest==6.0.1"],
    test_suite="tests",
)