from setuptools import setup, find_packages

VERSION = '1.0'


def readme():
    with open('README.md', 'r') as f:
        return f.read()


requirements = [
    'numpy>=2.1.0',
    'pandas>=2.2.2',
    'scikit-learn>=1.5.1',
    'scipy>=1.14.0',
    'setuptools>=73.0.1',
    'joblib>=1.4.2'
]


setup(
    name='LogNNet',
    version=VERSION,
    description='This package implements the lognet neural network, which uses chaotic transformations in the '
                'weight matrix to more effectively recognize patterns in data. With the ability to generate '
                'weight coefficients on the fly from only four parameters, it is effective on devices with '
                'limited resources and is suitable for classification, regression, time series '
                'forecasting, and other areas.',
    long_description=readme(),
    long_description_content_type="text/markdown",
    url='https://github.com/izotov93/LogNNet',
    author="Yuriy Izotov and Andrei Velichko",
    author_email='izotov93@yandex.ru',
    install_requires=requirements,
    packages=find_packages(),
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.11',
        'License :: OSI Approved :: MIT License',
    ],
    python_requires='>=3.11',
    include_package_data=True,
)
