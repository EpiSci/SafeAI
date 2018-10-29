import setuptools

with open("README.md", 'r') as fh:
    long_description = fh.read()

setuptools.setup(
    name='safeai',
    version='0.0.2',
    author='Episys Science',
    author_email='sangsulee92@gmail.com',
    description='SafeAI',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/EpisysScience/SafeAI',
    packages=setuptools.find_packages(),
    classifiers=[
        'Programming Language :: Python :: 3 :: Only',
    ]
)
