# src/nithinlang/__init__.py
"""
NithinLang V1
=============
A revolutionary 100% Free, Open-Source, Zero-Cloud, Multi-Lingual,
Ultra-Fast Programming Language.

Features:
  - Multi-lingual syntax (Telugu, Hindi, English)
  - LLVM-backed JIT compilation via llvmlite
  - Built-in ML/DS engine (numpy, pandas, scikit-learn)
  - Built-in 2D Game Engine (pygame)
  - Zero-Cloud local AI (Ollama/Llama.cpp)
  - Native file handling, loops, conditionals, functions

Usage:
  nithin run <filename.nl>
  nithin check <filename.nl>
  nithin new <project_name>
  nithin version
"""

from __future__ import annotations

__version__      = "1.0.0"
__author__       = "NithinLang Contributors"
__license__      = "MIT"
__description__  = (
    "NithinLang V1: Free, Open-Source, Zero-Cloud, Multi-Lingual, "
    "Ultra-Fast Programming Language."
)

# Public re-exports
from nithinlang.compiler  import NithinCompiler
from nithinlang.parser    import NithinParser
from nithinlang.core_lib  import CoreLib

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "NithinCompiler",
    "NithinParser",
    "CoreLib",
]