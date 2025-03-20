from setuptools import setup, find_packages

setup(
    name="aiavatar",
    version="0.6.2",
    url="https://github.com/uezo/aiavatar",
    author="uezo",
    author_email="uezo@uezo.net",
    maintainer="uezo",
    maintainer_email="uezo@uezo.net",
    description="🥰 Building AI-based conversational avatars lightning fast ⚡️💬",
    # long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples*", "tests*"]),
    install_requires=["litests>=0.3.8", "numpy>=2.2.3", "PyAudio>=0.2.14"],
    license="Apache v2",
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
