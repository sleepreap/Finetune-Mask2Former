from setuptools import setup, find_namespace_packages

setup(name='segformer',
      python_requires=">=3.10",
      install_requires=[
          "transformers",
          "datasets",
          "evaluate",
          "lightning",
          "tqdm",
          "matplotlib",
          "opencv-python"
      ],
      )
