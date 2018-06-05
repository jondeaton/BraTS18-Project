from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
                     "tensorflow",
                     "tensorflow-gpu",
                     "numpy",
                     "matplotlib",
                     "pandas",
                     "nibabel",    # Neuro-Imaging file format library
                     "nipype",     # More neuro-imaging libraries
                     "nilearn",
                     
                     "itk",        # The next three are for reading BraTS15 dataset
                     "pydicom",
                     "medpy"]

setup(name='BraTS',
      version='0.1',
      packages=find_packages(),
      description="BraTS 2018 Challenge",
      author="Jon Deaton, Cam Backes",
      author_email="jdeaton@stanford.edu, cbackes@stanford.edu",
      license='MIT',
      install_requires=REQUIRED_PACKAGES,
      include_package_data=True,  # allows config file to be copied to GCP
      zip_safe=False)