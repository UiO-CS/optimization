from setuptools import setup, find_packages

setup(
    name='tfopt',
    version=0.1,
    author='Kristian Monsen Haug',
    author_email='krimha@math.uio.no',
    license='MIT',
    description='Optimization algorithms for solving optimization problems related to Compressive Sensing',
    url='https://github.com/UiO-CS/optimization',
    # install_requires=['tensorflow', 'numpy', 'tfwavelets'],
    # packages=['optimization'],
    packages=find_packages(),
    zip_safe=False)
