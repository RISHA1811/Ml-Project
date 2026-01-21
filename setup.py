from setuptools import find_packages, setup
from typing import List


def get_requirements(file_path:str)->List[str]:
    '''
    This function will return the list of requirements
    '''
    requirements=[]
    with open(file_path) as file_obj:
        requirements=file_obj.readlines()
        requirements=[req.replace("\n","") for req in requirements ]



setup(
    name='Ml-Project',
    version='0.0.1',
    author='Rishabh Jha',
    author_email='rishabhjha1811@gmail.com',
    packages=find_packages(),
    install_requires=get_requirements('requirements.txt')
)