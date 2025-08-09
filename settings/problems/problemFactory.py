from .problemBase import *
import pkgutil
import importlib


def find_subclasses(base_class, module_name='settings.problems'):
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

def create_instance(class_name, module_name='.'):
    module = importlib.import_module(module_name)
    cls = getattr(module, class_name)
    return cls()

def problemFactory(problemName:str, savePath='./data/settings/'):
    allProblemSubClass = find_subclasses(problemBase)

    # return if class.name is problemName
    for problemClass in allProblemSubClass:
        if problemName == problemClass.__name__:
            return problemClass()

    # get all ext->class map
    ext = 'npy'

    listDir = os.listdir(savePath)
    fileNameWithExt = problemName + '.' + ext
    if fileNameWithExt in listDir:
        return npyProblem(os.path.join(savePath, fileNameWithExt))

    assert 'No this kind of problem'



if __name__ == '__main__':
    # jsonproblem = npyProblem('data/settings/TipCantilever_90_30_30.npy')
    # print(jsonproblem)
    problem = problemFactory('TipCantilever_90_30_30')
    print(problem.serialize())