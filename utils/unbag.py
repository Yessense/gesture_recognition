import os
import subprocess
import sys
from pathlib import Path
from sys import stdout

retval = os.getcwd()
print("Current working directory %s" % retval)
dir_path = '/media/yessense/Transcend/dataset'
converted_path = 'media/yessense/Transcend/converted'
os.chdir('/')

for directory in os.listdir(dir_path):
    i = 0
    for bag in os.listdir(os.path.join(dir_path, directory)):
        if bag.endswith('.bag'):
            name = Path(bag).stem
            new_dir = os.path.join(converted_path, directory)

            os.makedirs(os.path.join(converted_path, directory, str(i)), exist_ok=True)
            try:
                cmd = f'rs-convert -i {os.path.join(dir_path, directory, bag)} -p {os.path.join(converted_path, directory, str(i), f"{directory}_{i}")}'
                os.system(cmd)
            except Exception as e:
                pass
            i += 1
