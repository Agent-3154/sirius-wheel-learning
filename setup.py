from setuptools import setup, find_packages

setup(
    name="sirius-wheel-learning",
    version="0.1.0",
    packages=find_packages(),
    entry_points={
        "active_adaptation.projects": [
            "sirius = sirius",
        ],
        "active_adaptation.learning": [
            "sirius = sirius_learning",
        ]
    },
    install_requires=[
        "active-adaptation",
    ],
)