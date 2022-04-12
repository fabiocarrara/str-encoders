from setuptools import setup

setup(
    name='surrogate',
    version='0.1',
    description='Surrogate Text Encoders for Real Vectors',
    url='https://github.com/fabiocarrara/str-encoders',
    author='@fabiocarrara',
    author_email='fabio.carrara@isti.cnr.it',
    license='MIT',
    packages=['surrogate'],
    install_requires=['scikit-learn'],
    zip_safe=False
)