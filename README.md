1. Make sure to have Python 3.10 or later installed
2. Download the source code
3. Unpack it
4. Navigate to the created "fastrometry-VERSION" folder using the terminal
5. Manually install the build dependencies by typing "pip install setuptools wheel Cython numpy" (make sure pip is up-to-date)
5. In the terminal, type "py setup.py bdist_wheel" (if on macOS or Linux, replace "py" with "python3.xx", where xx is your Python version)
6. Navigate downwards one step with "cd dist"
7. Type "pip install FILENAME.whl"
8. Move to the directory containing your fits images (if you plan to use fastrometry at the command line)
9. You're ready to use fastrometry. You can either call fastrometry at the command line or import it as a module into one of your python applications.