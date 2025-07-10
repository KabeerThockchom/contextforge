#!/usr/bin/env python3
"""Setup script for ContextForge package."""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    requirements = []
    if os.path.exists("requirements.txt"):
        with open("requirements.txt", "r", encoding="utf-8") as fh:
            requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]
    return requirements

setup(
    name="contextforge",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python package for building LLM applications with sophisticated context engineering",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/contextforge",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
        "docs": [
            "sphinx>=5.0.0",
            "sphinx-rtd-theme>=1.2.0",
        ],
    },
    package_data={
        "contextforge": ["py.typed"],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="llm, ai, context, engineering, openai, anthropic, ollama, rag, tools",
    project_urls={
        "Bug Reports": "https://github.com/yourusername/contextforge/issues",
        "Source": "https://github.com/yourusername/contextforge",
        "Documentation": "https://contextforge.readthedocs.io/",
    },
) 