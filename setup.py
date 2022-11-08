import io
import os
import re
from os import path

from setuptools import find_packages
from setuptools import setup


this_directory = path.abspath(path.dirname(__file__))
with open(path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()


setup(
    name="sd-server",
    version="0.1.0",
    url="https://github.com/shyamsn97/stable-diffusion-server",
    license='MIT',

    author="Shyam Sudhakaran",
    author_email="shyamsnair@protonmail.com",

    description="Stable Diffusion server built with FastAPI",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=find_packages(exclude=('tests',)),

    install_requires=[
        'fastapi',
        'pyngrok',
        'nest-asyncio',
        'uvicorn',
        'starlette==0.20.4',
        'pillow',
        'numpy',
        'diffusers',
        'transformers',
        'torch',
        'accelerate'
    ],

    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
)
