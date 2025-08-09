from settings.problems.problemFactory import problemFactory
from settings.materials.materialFactory import materialFactory

def problemMaterialFactory(args):
    problem, material = args.problem, args.material
    problem = problemFactory(problem)
    if len(material) > 0:
        _material = materialFactory(material) # instant / object
        if _material is not None:
            problem.materialProperty = _material
            # setattr(args, 'isotropic', _material.isotropic)
    return problem
