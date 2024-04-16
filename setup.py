from setuptools import setup, find_namespace_packages

setup(name='Mask2Former',
      python_requires=">=3.10",
      install_requires=[
          "transformers",
          "datasets",
          "evaluate",
          "lightning",
          "tqdm",
          "matplotlib",
          "opencv-python",
          "scipy"
      ],
      )
