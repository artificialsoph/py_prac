from setuptools import setup, find_packages

setup(
    name='pymatrix',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'Click',
        'pandas',
        'scipy',
        'numpy',
        'scikit-learn',
    ],
    entry_points='''
        [console_scripts]
        pymatrix=pymatrix:main
    ''',
)
