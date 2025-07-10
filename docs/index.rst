ContextForge Documentation
==========================

ContextForge is a Python package for building LLM applications with sophisticated context engineering.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api
   examples
   contributing

Features
--------

* **Context Engineering**: Sophisticated context management for LLM applications
* **Multiple Providers**: Support for OpenAI, Anthropic, and Ollama
* **Memory Management**: Persistent conversation memory with multiple storage backends
* **Tool Calling**: Function calling support for interactive applications
* **RAG Support**: Vector-based retrieval for enhanced context
* **Streaming**: Real-time response streaming for all providers
* **Type Safety**: Full type hints for better development experience
* **Async Support**: Built with async/await for high performance

Quick Start
-----------

.. code-block:: python

   from contextforge import ContextEngine

   # Initialize with OpenAI
   engine = ContextEngine("openai", api_key="your-api-key")

   # Generate a response
   response = await engine.generate("Hello, world!")
   print(response)

Installation
------------

.. code-block:: bash

   pip install contextforge

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 