# src/nithinlang/core_lib.py
"""
NithinLang Standard Library (CoreLib)
======================================
Provides all built-in functions injected into every NithinLang program:

  I/O         : nl_print, nl_input
  File        : f_open, f_read, f_write, f_close, f_append,
                f_exists, f_delete, f_lines
  Math        : math_sqrt, math_pow, math_abs, math_floor, math_ceil,
                math_round, math_log, math_sin, math_cos, math_tan
  Vectors     : vec_add, vec_sub, vec_mul, vec_dot
  Matrices    : mat_mul
  Timing      : timer_start, timer_stop, nl_sleep
  System      : nl_exit, nl_env, nl_args, nl_platform
  Random      : nl_random, nl_randint, nl_uuid
  Hash        : nl_hash
  JSON        : nl_json_load, nl_json_dump
  HTTP        : nl_http_get, nl_http_post
"""

from __future__ import annotations

import math
import time
import os
import sys
import json
import random
import hashlib
import platform
import uuid as _uuid
from typing import Any, Dict, IO, List, Optional, Union

# ---------------------------------------------------------------------------
# Optional imports (graceful degradation)
# ---------------------------------------------------------------------------
try:
    import numpy as np
    _NP_AVAILABLE = True
except ImportError:
    _NP_AVAILABLE = False

try:
    import requests
    _REQUESTS_AVAILABLE = True
except ImportError:
    _REQUESTS_AVAILABLE = False


# ---------------------------------------------------------------------------
# CoreLib class
# ---------------------------------------------------------------------------

class CoreLib:
    """
    Container for all NithinLang standard library functions.
    Every method is designed to be injected as a global in the exec namespace.
    """

    MATH_PI: float = math.pi
    MATH_E : float = math.e

    # ── Internal state ────────────────────────────────────────────────────

    def __init__(self) -> None:
        self._open_files : Dict[str, IO[Any]]     = {}
        self._timers     : Dict[str, float]        = {}
        self._rand       : random.Random           = random.Random()

    # =========================================================================
    # I / O
    # =========================================================================

    def nl_print(self, *args: Any, sep: str = " ", end: str = "\n") -> None:
        """
        NithinLang's print function.  Mirrors Python's print() but supports
        multi-value output with custom separators.

        Example (telugu):  raayi("namaste")
        Example (hindi):   likho("namaste")
        Example (english): print("namaste")
        """
        sys.stdout.write(sep.join(str(a) for a in args) + end)
        sys.stdout.flush()

    def nl_input(self, prompt: str = "") -> str:
        """
        Read a line of input from the user.

        Example: name = input("Mee peru: ")
        """
        return input(prompt)

    # =========================================================================
    # File Handling
    # =========================================================================

    def f_open(
        self,
        path : str,
        mode : str = "r",
        encoding: str = "utf-8",
    ) -> str:
        """
        Open a file and return a handle-id string.

        Args:
            path     : File path (absolute or relative).
            mode     : "r" read, "w" write, "a" append, "rb" read-binary, etc.
            encoding : Text encoding (default utf-8, ignored for binary modes).

        Returns:
            A handle-id string used by f_read / f_write / f_close.

        Example:
            fh = f_open("data.txt", "r")
            content = f_read(fh)
            f_close(fh)
        """
        handle_id = f"fh_{os.path.abspath(path)}_{mode}_{id(path)}"
        try:
            if "b" in mode:
                fh = open(path, mode)
            else:
                fh = open(path, mode, encoding=encoding)
            self._open_files[handle_id] = fh
        except OSError as exc:
            raise IOError(f"f_open: Cannot open '{path}': {exc}") from exc
        return handle_id

    def f_read(self, handle_id: str, size: int = -1) -> str:
        """
        Read from an open file.

        Args:
            handle_id : Handle returned by f_open.
            size      : Number of bytes/chars (-1 = read all).

        Returns:
            File content as string (or bytes for binary mode).
        """
        fh = self._get_handle(handle_id, "f_read")
        return fh.read(size) if size >= 0 else fh.read()

    def f_write(self, handle_id: str, content: Union[str, bytes]) -> int:
        """
        Write content to an open file.

        Returns:
            Number of characters/bytes written.
        """
        fh = self._get_handle(handle_id, "f_write")
        return fh.write(content)

    def f_append(self, path: str, content: str, encoding: str = "utf-8") -> None:
        """
        Append content to a file (opens and closes internally).

        Example:
            f_append("log.txt", "new line\n")
        """
        with open(path, "a", encoding=encoding) as fh:
            fh.write(content)

    def f_close(self, handle_id: str) -> None:
        """Close an open file handle."""
        fh = self._get_handle(handle_id, "f_close")
        fh.close()
        del self._open_files[handle_id]

    def f_lines(self, path: str, encoding: str = "utf-8") -> List[str]:
        """
        Read all lines of a text file as a list.

        Example:
            lines = f_lines("data.txt")
            for line in lines:
                print(line)
        """
        with open(path, "r", encoding=encoding) as fh:
            return fh.readlines()

    def f_exists(self, path: str) -> bool:
        """Return True if path exists on disk."""
        return os.path.exists(path)

    def f_delete(self, path: str) -> bool:
        """
        Delete a file.

        Returns:
            True on success, False if file did not exist.
        """
        if os.path.isfile(path):
            os.remove(path)
            return True
        return False

    def _get_handle(self, handle_id: str, caller: str) -> IO[Any]:
        fh = self._open_files.get(handle_id)
        if fh is None:
            raise IOError(
                f"{caller}: Invalid or closed file handle '{handle_id}'. "
                "Did you call f_open() first?"
            )
        return fh

    # =========================================================================
    # Mathematics
    # =========================================================================

    def math_sqrt(self, x: float) -> float:
        """Square root of x."""
        return math.sqrt(x)

    def math_pow(self, base: float, exp: float) -> float:
        """base raised to the power exp."""
        return math.pow(base, exp)

    def math_abs(self, x: float) -> float:
        """Absolute value of x."""
        return abs(x)

    def math_floor(self, x: float) -> int:
        """Floor of x."""
        return math.floor(x)

    def math_ceil(self, x: float) -> int:
        """Ceiling of x."""
        return math.ceil(x)

    def math_round(self, x: float, ndigits: int = 0) -> float:
        """Round x to ndigits decimal places."""
        return round(x, ndigits)

    def math_log(self, x: float, base: float = math.e) -> float:
        """Logarithm of x (default: natural log)."""
        return math.log(x, base)

    def math_sin(self, x: float) -> float:
        """Sine of x (radians)."""
        return math.sin(x)

    def math_cos(self, x: float) -> float:
        """Cosine of x (radians)."""
        return math.cos(x)

    def math_tan(self, x: float) -> float:
        """Tangent of x (radians)."""
        return math.tan(x)

    # =========================================================================
    # Vector / Matrix (uses numpy when available, pure Python fallback)
    # =========================================================================

    def vec_add(
        self,
        a: Union[List[float], "np.ndarray"],
        b: Union[List[float], "np.ndarray"],
    ) -> Union[List[float], "np.ndarray"]:
        """Element-wise vector addition."""
        if _NP_AVAILABLE:
            return np.add(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
        return [x + y for x, y in zip(a, b)]

    def vec_sub(
        self,
        a: Union[List[float], "np.ndarray"],
        b: Union[List[float], "np.ndarray"],
    ) -> Union[List[float], "np.ndarray"]:
        """Element-wise vector subtraction."""
        if _NP_AVAILABLE:
            return np.subtract(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
        return [x - y for x, y in zip(a, b)]

    def vec_mul(
        self,
        a: Union[List[float], "np.ndarray"],
        scalar: float,
    ) -> Union[List[float], "np.ndarray"]:
        """Multiply each element of a by scalar."""
        if _NP_AVAILABLE:
            return np.asarray(a, dtype=np.float64) * scalar
        return [x * scalar for x in a]

    def vec_dot(
        self,
        a: Union[List[float], "np.ndarray"],
        b: Union[List[float], "np.ndarray"],
    ) -> float:
        """Dot product of two vectors."""
        if _NP_AVAILABLE:
            return float(np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))
        return sum(x * y for x, y in zip(a, b))

    def mat_mul(
        self,
        a: Union[List[List[float]], "np.ndarray"],
        b: Union[List[List[float]], "np.ndarray"],
    ) -> Union[List[List[float]], "np.ndarray"]:
        """Matrix multiplication."""
        if _NP_AVAILABLE:
            return np.matmul(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64))
        # Pure Python fallback
        rows_a = len(a)
        cols_a = len(a[0])
        cols_b = len(b[0])
        result = [[0.0] * cols_b for _ in range(rows_a)]
        for i in range(rows_a):
            for j in range(cols_b):
                for k in range(cols_a):
                    result[i][j] += a[i][k] * b[k][j]
        return result

    # =========================================================================
    # Timing
    # =========================================================================

    def timer_start(self, label: str = "default") -> None:
        """Start a named timer."""
        self._timers[label] = time.perf_counter()

    def timer_stop(self, label: str = "default") -> float:
        """
        Stop a named timer and return elapsed seconds.

        Example:
            timer_start("loop")
            for i in range(1_000_000):
                pass
            elapsed = timer_stop("loop")
            print(elapsed)
        """
        if label not in self._timers:
            raise RuntimeError(
                f"timer_stop: Timer '{label}' was never started. "
                "Call timer_start() first."
            )
        elapsed = time.perf_counter() - self._timers.pop(label)
        return elapsed

    def nl_sleep(self, seconds: float) -> None:
        """Pause execution for the given number of seconds."""
        time.sleep(seconds)

    # =========================================================================
    # System / Environment
    # =========================================================================

    def nl_exit(self, code: int = 0) -> None:
        """Terminate the NithinLang program with the given exit code."""
        sys.exit(code)

    def nl_env(self, name: str, default: str = "") -> str:
        """Read an environment variable."""
        return os.environ.get(name, default)

    def nl_args(self) -> List[str]:
        """Return the list of command-line arguments."""
        return sys.argv[:]

    def nl_platform(self) -> str:
        """Return a string identifying the current operating system."""
        return platform.system()

    # =========================================================================
    # Random
    # =========================================================================

    def nl_random(self) -> float:
        """Return a random float in [0.0, 1.0)."""
        return self._rand.random()

    def nl_randint(self, low: int, high: int) -> int:
        """Return a random integer N such that low <= N <= high."""
        return self._rand.randint(low, high)

    def nl_uuid(self) -> str:
        """Return a new random UUID4 string."""
        return str(_uuid.uuid4())

    # =========================================================================
    # Hashing
    # =========================================================================

    def nl_hash(self, text: str, algo: str = "sha256") -> str:
        """
        Compute a hex-digest hash of text.

        Args:
            text : Input string.
            algo : One of "md5", "sha1", "sha256", "sha512".

        Returns:
            Lowercase hex digest string.
        """
        try:
            h = hashlib.new(algo)
        except ValueError:
            raise ValueError(
                f"nl_hash: Unknown algorithm '{algo}'. "
                "Choose from: md5, sha1, sha256, sha512"
            )
        h.update(text.encode("utf-8"))
        return h.hexdigest()

    # =========================================================================
    # JSON
    # =========================================================================

    def nl_json_load(self, text: str) -> Any:
        """Parse a JSON string and return a Python object."""
        return json.loads(text)

    def nl_json_dump(self, obj: Any, indent: int = 2) -> str:
        """Serialise a Python object to a JSON string."""
        return json.dumps(obj, indent=indent, ensure_ascii=False)

    # =========================================================================
    # HTTP (requires `requests`)
    # =========================================================================

    def nl_http_get(
        self,
        url     : str,
        headers : Optional[Dict[str, str]] = None,
        timeout : int = 30,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP GET request.

        Returns:
            {"status": int, "body": str, "headers": dict}
        """
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError(
                "nl_http_get: 'requests' package is not installed. "
                "Run: pip install requests"
            )
        resp = requests.get(url, headers=headers or {}, timeout=timeout)
        return {
            "status"  : resp.status_code,
            "body"    : resp.text,
            "headers" : dict(resp.headers),
        }

    def nl_http_post(
        self,
        url     : str,
        data    : Optional[Dict[str, Any]] = None,
        json_   : Optional[Any]            = None,
        headers : Optional[Dict[str, str]] = None,
        timeout : int = 30,
    ) -> Dict[str, Any]:
        """
        Perform an HTTP POST request.

        Returns:
            {"status": int, "body": str, "headers": dict}
        """
        if not _REQUESTS_AVAILABLE:
            raise RuntimeError(
                "nl_http_post: 'requests' package is not installed. "
                "Run: pip install requests"
            )
        resp = requests.post(
            url,
            data    = data,
            json    = json_,
            headers = headers or {},
            timeout = timeout,
        )
        return {
            "status"  : resp.status_code,
            "body"    : resp.text,
            "headers" : dict(resp.headers),
        }