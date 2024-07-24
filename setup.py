from setuptools import setup, find_packages

setup(
    name='barebonesnn',
    version='0.1.1',
    packages=find_packages(),
    description='A neural network library built from scratch.',
    author='Amir Mohammadikarbalaei',
    author_email='a.mohammadikarbalaei@gmail.com',
    url='https://github.com/AmirMohammadiKarbalaei/BareBonesNN/barebonesnn',
    install_requires=[
        'numpy>=1.21.0', 
    ],
    python_requires='>=3.6',
    long_description=open("README_pypi.md").read(),
)