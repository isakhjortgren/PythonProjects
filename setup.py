from distutils.core import setup
import os
from Cython.Build import cythonize


list_of_cython_files = ['fib.py']


for cython_file in list_of_cython_files:
    pyx_file_name = 'c' + cython_file + 'x'
    os.rename(cython_file, pyx_file_name)
    setup(
        ext_modules=cythonize(pyx_file_name)
    )
    os.rename(pyx_file_name, cython_file)
