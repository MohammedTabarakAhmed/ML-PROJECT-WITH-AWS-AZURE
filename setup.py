from setuptools import find_packages, setup
from typing import List

def get_requirements(file_path: str) -> List[str]:
    """
    This function returns the list of requirements from requirements.txt
    """
    requirements = []
    with open(file_path) as file_obj:
        requirements = file_obj.readlines()
        # Clean each line (remove newline and spaces)
        requirements = [req.strip() for req in requirements if req.strip()]
        
        # Remove editable install references like '-e .' if present
        editable = "-e ."
        if editable in requirements:
            requirements.remove(editable)
        
        return requirements

setup(
    name='ML_proj',
    version='0.0.1',
    author='Ahmed',
    author_email='tabarakahmed030@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)
