import os

import setuptools
from setuptools import setup

def version():
  initPath = os.path.abspath(os.path.join(__file__, "..", "deepEMhancer", "__init__.py"))
  with open(initPath) as f:
    version = f.read().strip().split('"')[-2]
  return version
      
def readme():
  readmePath = os.path.abspath(os.path.join(__file__, "..", "README.md"))
  try:
    with open(readmePath) as f:
      return f.read()
  except UnicodeDecodeError:
    try:
      with open(readmePath, 'r', encoding='utf-8') as f:
        return f.read()
    except Exception as e:
      return "Description not available due to unexpected error: "+str(e)


install_requires = [
  'numpy>=1.24,<2.0',
  'scikit-image>=0.22',
  'scipy>=1.14',
  'joblib>=1.4',
  'mrcfile>=1.4',
  'requests>=2.32',
  'tqdm>=4.66',
]

installTfCpuOnly = os.environ.get("DEEPEMHANCER_CPU_ONLY", None)
if not installTfCpuOnly:
  tfTarget='tensorflow[and-cuda]>=2.18'
else:
  tfTarget='tensorflow>=2.18'
install_requires.append(tfTarget)

setup(name='deepEMhancer',
      version=version(),
      description='Deep learning for cryo-EM maps post-processing',
      long_description=readme(),
      long_description_content_type="text/markdown",
      keywords='cryo-EM deep learning',
      url='https://github.com/rsanchezgarc/deepEMhancer',
      author='Ruben Sanchez-Garcia',
      author_email='rsanchez@cnb.csic.es',
      license='Apache 2.0',
      packages=setuptools.find_packages(),
      install_requires=install_requires,
      dependency_links=[],
      entry_points={
        'console_scripts': ['deepemhancer=deepEMhancer.exeDeepEMhancer:commanLineFun'],
      },
      include_package_data=True,
      zip_safe=False)
#python -c "import tensorflow as tf; tf.zeros((3,2))" && python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"


