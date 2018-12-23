import importlib
import glob
module_names = glob.glob('./rcnn/symbols/*.py')
for module_name in module_names:
    name = module_name.split('/')[-1].split('.')[0]
    if name == '__init__':
        continue
    importlib.import_module('symbols.' + name)