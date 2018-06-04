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
          "tensorflow-gpu",
          "numpy",
          "matplotlib",
          "pandas",
          "nibabel",    # Neuro-Imaging file format library
          "nipype",     # More neuro-imaging libraries
          "nilearn",
          "SimpleITK"   # For normalizing the data
                        
          "itk",        # The next three are for reading BraTS15 dataset
          "pydicom",
          "medpy",
          "keras"
      ],
      zip_safe=False)
