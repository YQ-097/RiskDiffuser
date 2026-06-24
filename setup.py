import os

import setuptools

# Change directory to allow installation from anywhere
script_folder = os.path.dirname(os.path.realpath(__file__))
os.chdir(script_folder)

# Installs
setuptools.setup(
    name="riskdiffuser",
    version="1.0.0",
    author="Y. Qian, Z. Wang, Y. Wu, et al.",
    description=(
        "RiskDiffuser: Continuous risk conditioning for diffusion planning "
        "in autonomous driving"
    ),
    packages=setuptools.find_packages(),
    package_dir={"": "."},
    classifiers=[
        "Programming Language :: Python :: 3.9",
        "Operating System :: OS Independent",
    ],
)
