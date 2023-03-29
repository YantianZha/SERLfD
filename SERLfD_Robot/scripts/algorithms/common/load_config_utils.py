from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

def loadYAML(yamlFile):
    with open(yamlFile) as file:
        data = load(file, Loader=Loader)
        output_str = dump(data, Dumper=Dumper)
    return output_str, data