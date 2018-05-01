from setuptools import setup, find_packages

setup(name='BraTS',
      version='0.1',
      packages=find_packages(),
      description="BraTS 2018 Challenge",
      author="Jon Deaton, Cam Backes",
      author_email="jdeaton@stanford.edu, cbackes@stanford.edu",
      license='MIT',
      install_requires=[
          "tensorflow",
          "numpy",
          "matplotlib",
          "nibabel"        # Neuro-Imaging file format library
      ],
      zip_safe=False)