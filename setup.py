from setuptools import setup, find_packages

setup(
    name="databench_eval",
    version="3.0.1",
    author="jorses",
    author_email="jorgeosesgrijalba@gmail.com",
    description="Evaluation framework for DataBench",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/jorses/databench",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    project_urls={
        "Homepage": "https://github.com/jorses/databench",
        "Issues": "https://github.com/jorses/databench/issues",
    },
    install_requires=[
        "datasets",
        "tqdm",
        "pandas",
        "numpy",
        # Add other dependencies here
    ],
)
