from setuptools import setup, find_packages
setup(
    name="DarkElf",
    version="1.1",
    package_dir = {"darkelf":"darkelf"},
    packages=find_packages(),
    author='Brian Campbell-Deem, Simon Knapen, Jonathan Kozaczuk, Tongyan Lin, Connor Stratman, and Ethan Villarama',
    description="package capable of calculating interaction rates of light dark matter in dielectric materials, including screening effects",

)
