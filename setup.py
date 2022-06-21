from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy as np

#Because "py -m build" doesn't seem to build directly from pyx files without errors, we have to first run "py setup.py build_ext --inplace"
#to generate c sources, which can then be used by cibuildwheel to build linux wheels (cibuildwheel seems to implicity call "py -m build"
#rather than "py setup.py sdist bdist_wheel", which can be used for building regular windows wheels directly from pyx files).

extensions_pyx = [
    Extension("fastrometry.cython_code.PSE",["src/fastrometry/cython_code/PSE.pyx"]),
    Extension("fastrometry.cython_code.WCS",["src/fastrometry/cython_code/WCS.pyx"])
    ]

extensions_c = [
    Extension("fastrometry.cython_code.PSE",["src/fastrometry/cython_code/PSE.c"], language='c'),
    Extension("fastrometry.cython_code.WCS",["src/fastrometry/cython_code/WCS.c"], language='c')
    ]

def my_cythonize():
    try:
        return cythonize(extensions_pyx)
    except:
        return cythonize(extensions_c)

setup(
    ext_modules = my_cythonize(),
    zip_safe=False,
    include_dirs = [np.get_include()],

    project_urls = {
        'Homepage': 'https://github.com/user29A/fastrometry/wiki',
        'Download': 'https://pypi.org/project/fastrometry/#files',
        'Source code': 'https://github.com/user29A/fastrometry',
        'Bug tracker': 'https://github.com/user29A/fastrometry/issues',
        'Documentation': 'https://github.com/user29A/fastrometry/wiki'
        }
)