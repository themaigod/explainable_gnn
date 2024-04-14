from setuptools import setup

setup(
    name='explainable_gnn',
    version='0.0.1',
    packages=['explainable_gnn'],
    author='JIANG Maiqi',
    author_email='jiangmaiqi2271108464@gmail.com',
    description='A package for explainable graph neural networks',
    long_description='A package for explainable graph neural networks, further improving the speed of the inference process',
    # url=
    # download_url=
    keywords=['graph neural networks', 'explainable', 'pytorch', 'dgl', 'pyg'],
    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    install_requires=[
        'torch',
        'numpy',
        'scipy',
    ],
)

