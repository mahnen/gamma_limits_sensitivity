from setuptools import setup

setup(
    name='gamma_limits_sensitivity',
    version='0.1',
    description='Calculate integral UL and sensitiv. for g-ray telescopes.',
    url='https://github.com/mahnen/gamma_limits_sensitivity',
    author='Max Ahnen',
    author_email='m.knoetig@gmail.com',
    licence='MIT',
    packages=[
        'gamma_limits_sensitivity'
    ],
    package_data={'gamma_limits_sensitivity': ['resources/A_eff/*']},
    entry_points={
        'console_scripts': [
            'gamma_limits_sensitivity = gamma_limits_sensitivity.__main__:main'
        ]
    },
    install_requires={
        'numpy',
        'scipy',
        'matplotlib',
        'docopt'
    },
    tests_require=['pytest']
)
