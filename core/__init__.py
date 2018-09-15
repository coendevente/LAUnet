import inspect
import os

# Make the parent folder of the folder in which this file is located, the main path. This is done to make sure that when
# the file with the main function is located outside of this folder, the main path is again set to where all the
# relative paths are intended to be relative to.
os.chdir('/'.join(inspect.stack()[0][1].split('/')[:-2]))
