import os
import subprocess
import sys
from pathlib import Path
from sys import stdout

retval = os.getcwd()
converted_path = 'media/yessense/Transcend/converted'

os.chdir('/')
print("Current working directory %s" % retval)

for root, dirs, files in os.walk(converted_path):
    for file in files:
        if not file.endswith('.png'):
            file_path = os.path.join(root,file)
            os.remove(file_path)
