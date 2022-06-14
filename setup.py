from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [Extension("src.fastrometry.cython_code.*",["src\\fastrometry\\cython_code\\*.pyx"])]

setup(
    ext_modules = cythonize(extensions),
    zip_safe=False,
    include_dirs = [numpy.get_include()],
)