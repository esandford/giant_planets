# to run: python setup.py build_ext --inplace                                                                                                                                                                  
#from setuptools import setup,Extension                                                                                                                                                                        
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

extensions = [Extension('EOSutils.EOSutils',['EOSutils/EOSutils.py'])]

setup(name='EOSutils',
      version='0.0',
      description='',
      author='Emily Sandford',
      author_email='sandford@strw.leidenuniv.nl',
      url='',
      license='MIT',
      packages=['EOSutils'],
      include_dirs=[np.get_include()])
      #install_requires=['numpy','matplotlib','warnings','scipy','copy','math','itertools','collections'],                                                                                                     
      #ext_modules=cythonize(extensions))   
