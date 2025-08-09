from .materialsBase import *
import inspect
import importlib
import pkgutil

def find_subclasses(base_class, module_name='settings.materials'):
    subclasses = []
    package = importlib.import_module(module_name)
    package_path = package.__path__

    for _, name, is_pkg in pkgutil.iter_modules(package_path):
        if not is_pkg:
            module = importlib.import_module(f"{module_name}.{name}")
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and obj is not base_class:
                    subclasses.append(obj)
    return subclasses


def materialFactory(materialName:str):
    allMaterialSubClass = find_subclasses(materialBase)
    for materialClass in allMaterialSubClass:
        if materialName == materialClass.__name__:
            return materialClass()

    assert 'No this kind of material'