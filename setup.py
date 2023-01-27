from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(name='nmf_methods',
      version='0.35',
      description='Python implementation of Nonnegative Matrix Factorization methods.',
      url='https://github.com/waqasbinhamed/nmf_methods',
      author='Waqas Bin Hamed',
      author_email='waqasbinhamed@gmail.com',
      packages=find_packages(),
      python_requires='>=3.6.0',
      install_requires=['numpy', 'scipy']
      )
