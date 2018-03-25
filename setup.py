try:
    from setuptools import setup, find_packages
except ImportError :
    raise ImportError("setuptools module required, please go to "
                      "https://pypi.python.org/pypi/setuptools and '"
                      "follow the instructions for installing setuptools")

desc = "A fast implementation of GloVe, with optional retrofitting to " \
       "pre-existing GloVe vectors. "

setup(
    name="mittens",
    version="0.1",
    description="Fast GloVe with optional retrofitting.",
    long_description=desc,
    author="Nick Dingwall, Chris Potts",
    author_email="nick/chris@roamanalytics.com",
    url="https://github.com/roamanalytics/mittens",
    packages=find_packages(exclude=["test"]),
    install_requires=[
        "numpy"
    ],
    test_suite="mittens/test",
)
