from setuptools import setup, find_packages

setup(
    name="aiavatar",
    version="0.5.0",
    url="https://github.com/uezo/aiavatar",
    author="uezo",
    author_email="uezo@uezo.net",
    maintainer="uezo",
    maintainer_email="uezo@uezo.net",
    description="🥰 Building AI-based conversational avatars lightning fast ⚡️💬",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=["examples*", "tests*"]),
    install_requires=["aiohttp==3.9.3", "numpy==1.24.3", "openai==1.12.0", "sounddevice==0.4.6"],
    license="Apache v2",
    classifiers=[
        "Programming Language :: Python :: 3"
    ]
)
