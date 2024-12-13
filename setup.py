from setuptools import setup, find_packages
import setuptools
#import re
#import torch
#
#import warnings
#warnings.filterwarnings("ignore")
#
## Disable version normalization performed by setuptools.setup()
#try:
#    # Try the approach of using sic(), added in setuptools 46.1.0
#    from setuptools import sic
#except ImportError:
#    # Try the approach of replacing packaging.version.Version
#    sic = lambda v: v
#    try:
#        # setuptools >=39.0.0 uses packaging from setuptools.extern
#        from setuptools.extern import packaging
#    except ImportError:
#        # setuptools <39.0.0 uses packaging from pkg_resources.extern
#        from pkg_resources.extern import packaging
#    packaging.version.Version = packaging.version.LegacyVersion
#
#with open("requirements.txt") as f:
#    required = f.read().splitlines()
#    if torch.cuda.is_available() is False:
#        torch_required = next(filter(lambda req: re.match('torch==', req) is not None, required))
#        cuda_plus_index = torch_required.index('+cu')
#        torch_required = torch_required[:cuda_plus_index]
#        required_except = list(filter(lambda req: re.match('torch==', req) is None, required))
#        required = required_except + [torch_required]

setup(
    name="tweak",
    version=sic("0.2.1"),
    url="https://github.com/cloudjo21/tweak.git",
    packages=find_packages("src"),
    package_dir={"tweak": "src/tweak"},
    python_requires=">=3.11.6",
    long_description=open("README.md").read(),
    install_requires=required,
    # normalize_version=False,
    dependency_links=[
        'https://download.pytorch.org/whl/torch',
    ]
)

