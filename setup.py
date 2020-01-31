from setuptools import setup

setup(name='wind-power-forecasting',
      description='',
      long_description='',
      version='0.0.1',
      author='Thomas Karaouzene',
      author_email = '"Thomas Karaouzene" <thomas.karaouzene@non-se>',
      url='https://github.com/tkaraouzene/wind_power_forecasting',
      licence='',
      package_dir={'wind-power-forecasting': 'wind_power_forecasting'},
      packages=['wpf'],
      install_requires=['pandas', 'numpy', 'matplotlib']
      )
