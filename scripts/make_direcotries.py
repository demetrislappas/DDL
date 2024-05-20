# Import native packages
import os

def md(path):
    delimeter = '/' if '/' in path else '\\'
    dir_list = path.split(delimeter)
    dir_list = dir_list[:-1] if '.' in dir_list[-1] else dir_list
    dir_path = delimeter.join(dir_list)
    if os.path.isdir(dir_path) == False:
        os.makedirs(dir_path)
    return path
