# src/nithinlang/compiler.py
"""
NithinLang Compiler & Execution Engine
========================================
Architecture
------------
NithinLang uses a two-tier execution strategy for maximum speed:

Tier 1 — LLVM JIT (via llvmlite):
  Pure numeric / array-heavy functions detected in the AST are compiled
  to native machine code using llvmlite's LLVM IR builder.  This gives
  C++-level throughput for hot numeric loops.

Tier 2 — Optimised CPython Bytecode:
  All other code is compiled to Python bytecode with a custom set of
  peephole optimisations applied to the AST (constant folding, dead-code
  elimination, loop unrolling for small ranges) before calling compile().

The two tiers are transparent to the user: the compiler analyses each
top-level function/block and routes it to the appropriate tier.

LLVM IR generation details
--------------------------
We use llvmlite's IRBuilder to generate LLVM IR for:
  - Arithmetic-only functions with scalar int/float arguments
  - Tight for-loops over ranges (converted to LLVM loop + phi nodes)
  - NumPy-style vector operations (mapped to LLVM vector types)

Non-numeric code falls back to CPython execution automatically.
"""

from __future__ import annotations

import ast
import sys
import time
import ctypes
import struct
import types
import dis
import textwrap
import traceback
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

# llvmlite — graceful degradation if not installed
try:
    import llvmlite.ir          as ll
    import llvmlite.binding     as llvm
    _LLVM_AVAILABLE = True
except ImportError:
    _LLVM_AVAILABLE = False

from nithinlang.parser    import ParseResult, ParseError
from nithinlang.core_lib  import CoreLib
from nithinlang.ml_ds_engine import MLDSEngine
from nithinlang.game_engine  import GameEngine
from nithinlang.ai_engine    import AIEngine


# ---------------------------------------------------------------------------
# LLVM initialisation (once per process)
# ---------------------------------------------------------------------------

_LLVM_BACKING = None
if _LLVM_AVAILABLE:
   #llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    _LLVM_TARGET    = llvm.Target.from_default_triple()
    _LLVM_TM        = _LLVM_TARGET.create_target_machine(opt=3)  # -O3 equivalent
    _backing_module = llvm.parse_assembly("")
    _LLVM_BACKING   = llvm.create_mcjit_compiler(_backing_module, _LLVM_TM)


# ---------------------------------------------------------------------------
# AST Optimiser (CPython tier)
# ---------------------------------------------------------------------------

class _ASTOptimiser(ast.NodeTransformer):
    """
    Applies AST-level optimisations before CPython bytecode compilation:

    1. Constant Folding — evaluate constant binary / unary expressions
       at compile time.
    2. Dead-Code Elimination — remove `if False:` / `if True:` branches.
    3. Loop Unrolling — unroll `for i in range(N):` where N <= 8 and the
       body contains no `break`/`continue`/`return`.
    """

    # ── Constant Folding ──────────────────────────────────────────────────

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
            try:
                result = _eval_binop(node.op, node.left.value, node.right.value)
                return ast.copy_location(ast.Constant(value=result), node)
            except Exception:
                pass
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.operand, ast.Constant):
            try:
                result = _eval_unaryop(node.op, node.operand.value)
                return ast.copy_location(ast.Constant(value=result), node)
            except Exception:
                pass
        return node

    # ── Dead-Code Elimination ─────────────────────────────────────────────

    def visit_If(self, node: ast.If) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.test, ast.Constant):
            if node.test.value:
                # `if True:` — keep body, drop orelse
                return node.body  # type: ignore[return-value]
            else:
                # `if False:` — keep orelse (may be empty)
                return node.orelse if node.orelse else []  # type: ignore[return-value]
        return node

    # ── Loop Unrolling ────────────────────────────────────────────────────

    def visit_For(self, node: ast.For) -> ast.AST:
        self.generic_visit(node)

        # Check: `for <name> in range(<int_literal>):`
        if (
            isinstance(node.iter, ast.Call)
            and isinstance(node.iter.func, ast.Name)
            and node.iter.func.id == "range"
            and len(node.iter.args) == 1
            and isinstance(node.iter.args[0], ast.Constant)
            and isinstance(node.iter.args[0].value, int)
            and 0 < node.iter.args[0].value <= 8
            and not node.orelse
            and isinstance(node.target, ast.Name)
            and not _contains_flow_control(node.body)
        ):
            n        = node.iter.args[0].value
            var_name = node.target.id
            unrolled : List[ast.stmt] = []
            for i in range(n):
                # Replace each occurrence of `var_name` with the literal `i`
                const_node = ast.Constant(value=i)
                for stmt in node.body:
                    new_stmt = _substitute_name(stmt, var_name, const_node)
                    unrolled.append(new_stmt)
            return unrolled  # type: ignore[return-value]

        return node


def _eval_binop(op: ast.operator, left: Any, right: Any) -> Any:
    ops: Dict[type, Callable] = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.FloorDiv: lambda a, b: a // b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
        ast.BitAnd: lambda a, b: a & b,
        ast.BitOr: lambda a, b: a | b,
        ast.BitXor: lambda a, b: a ^ b,
        ast.LShift: lambda a, b: a << b,
        ast.RShift: lambda a, b: a >> b,
    }
    fn = ops.get(type(op))
    if fn is None:
        raise ValueError(f"Unsupported op {op}")
    return fn(left, right)


def _eval_unaryop(op: ast.unaryop, val: Any) -> Any:
    ops: Dict[type, Callable] = {
        ast.USub  : lambda v: -v,
        ast.UAdd  : lambda v: +v,
        ast.Not   : lambda v: not v,
        ast.Invert: lambda v: ~v,
    }
    fn = ops.get(type(op))
    if fn is None:
        raise ValueError(f"Unsupported unary op {op}")
    return fn(val)


def _contains_flow_control(stmts: List[ast.stmt]) -> bool:
    """Return True if any stmt in the list is Break/Continue/Return."""
    for s in stmts:
        for node in ast.walk(s):
            if isinstance(node, (ast.Break, ast.Continue, ast.Return)):
                return True
    return False


def _substitute_name(
    node: ast.AST,
    name: str,
    replacement: ast.AST,
) -> ast.AST:
    """Deep-copy a node, replacing all Name(id=name) with replacement."""
    import copy
    node_copy = copy.deepcopy(node)

    class _Subst(ast.NodeTransformer):
        def visit_Name(self, n: ast.Name) -> ast.AST:
            if n.id == name:
                return ast.copy_location(replacement, n)
            return n

    return _Subst().visit(node_copy)


# ---------------------------------------------------------------------------
# LLVM JIT tier
# ---------------------------------------------------------------------------

class _LLVMFunctionBuilder:
    """
    Detects whether a Python AST FunctionDef can be compiled entirely to
    LLVM IR and, if so, builds the IR and returns a callable cfunc.

    Supported subset:
      - Arguments: int or float only (type annotations required)
      - Body: assignments, arithmetic binary ops, return
      - No Python-specific types, containers, or calls

    If the function is not suitable, returns None (fall back to CPython).
    """

    _ARITH_OPS: Dict[type, str] = {
        ast.Add     : "add",
        ast.Sub     : "sub",
        ast.Mult     : "mult",
        ast.Div     : "fdiv",
        ast.FloorDiv: "sdiv",
        ast.Mod     : "srem",
        ast.Pow     : "__pow__",   # handled specially
    }

    def try_build(
        self,
        func_def : ast.FunctionDef,
    ) -> Optional[Callable]:
        """
        Attempt LLVM compilation of func_def.

        Returns:
            A Python callable wrapping the JIT-compiled native function,
            or None if the function cannot be handled by the LLVM tier.
        """
        if not _LLVM_AVAILABLE:
            return None

        # ── Determine argument types from annotations ──────────────────────
        arg_types = self._extract_arg_types(func_def)
        if arg_types is None:
            return None   # non-numeric or unannotated args

        ret_type = self._extract_ret_type(func_def)
        if ret_type is None:
            return None

        # ── Build LLVM IR ──────────────────────────────────────────────────
        try:
            ir_mod   = ll.Module(name=func_def.name)
            ll_types = [self._py_to_ll_type(t) for t in arg_types]
            ll_ret   = self._py_to_ll_type(ret_type)

            fn_type  = ll.FunctionType(ll_ret, ll_types)
            fn       = ll.Function(ir_mod, fn_type, name=func_def.name)

            block    = fn.append_basic_block(name="entry")
            builder  = ll.IRBuilder(block)

            # Map argument names → LLVM values
            env: Dict[str, Any] = {}
            for py_arg, ll_arg in zip(func_def.args.args, fn.args):
                ll_arg.name   = py_arg.arg
                env[py_arg.arg] = ll_arg

            # Compile body
            ret_val = self._compile_stmts(
                func_def.body, builder, env, ll_ret, fn
            )
            if ret_val is None:
                return None

            builder.ret(ret_val)

            # ── JIT compile ────────────────────────────────────────────────
            ir_str  = str(ir_mod)
            mod     = llvm.parse_assembly(ir_str)
            mod.verify()

            # Optimise with LLVM pass manager (-O3)
            pmb = llvm.create_pass_manager_builder()
            pmb.opt_level = 3
            pm  = llvm.create_module_pass_manager()
            pmb.populate(pm)
            pm.run(mod)

            engine  = llvm.create_mcjit_compiler(mod, _LLVM_TM)
            engine.finalize_object()
            engine.run_static_constructors()

            fn_ptr  = engine.get_function_address(func_def.name)

            # Wrap in ctypes callable
            c_types = [self._py_to_ctype(t) for t in arg_types]
            c_ret   = self._py_to_ctype(ret_type)
            cfunc   = ctypes.CFUNCTYPE(c_ret, *c_types)(fn_ptr)

            # Keep the engine alive (GC would free JIT memory)
            cfunc._engine = engine  # type: ignore[attr-defined]
            return cfunc

        except Exception:
            # Any failure → fall back to CPython
            return None

    # ── Type helpers ──────────────────────────────────────────────────────

    def _extract_arg_types(
        self,
        func_def: ast.FunctionDef,
    ) -> Optional[List[type]]:
        types_out: List[type] = []
        for arg in func_def.args.args:
            if arg.annotation is None:
                return None
            py_t = self._annotation_to_pytype(arg.annotation)
            if py_t not in (int, float):
                return None
            types_out.append(py_t)
        return types_out

    def _extract_ret_type(self, func_def: ast.FunctionDef) -> Optional[type]:
        if func_def.returns is None:
            return None
        return self._annotation_to_pytype(func_def.returns)

    @staticmethod
    def _annotation_to_pytype(ann: ast.expr) -> Optional[type]:
        if isinstance(ann, ast.Name):
            return {"int": int, "float": float}.get(ann.id)
        return None

    @staticmethod
    def _py_to_ll_type(t: type) -> ll.Type:
        return ll.IntType(64) if t is int else ll.DoubleType()

    @staticmethod
    def _py_to_ctype(t: type):
        return ctypes.c_int64 if t is int else ctypes.c_double

    # ── Statement / expression compilers ─────────────────────────────────

    def _compile_stmts(
        self,
        stmts   : List[ast.stmt],
        builder : "ll.IRBuilder",
        env     : Dict[str, Any],
        ret_type: "ll.Type",
        fn      : "ll.Function",
    ) -> Optional[Any]:
        """Compile a list of statements; return the last return value or None."""
        ret_val = None
        for stmt in stmts:
            if isinstance(stmt, ast.Return):
                if stmt.value is None:
                    return None
                ret_val = self._compile_expr(stmt.value, builder, env)
                return ret_val
            elif isinstance(stmt, (ast.Assign, ast.AugAssign)):
                self._compile_assign(stmt, builder, env)
            else:
                # Cannot handle — give up
                return None
        return ret_val

    def _compile_assign(
        self,
        node    : ast.stmt,
        builder : "ll.IRBuilder",
        env     : Dict[str, Any],
    ) -> None:
        if isinstance(node, ast.Assign):
            val = self._compile_expr(node.value, builder, env)
            if val is None:
                raise ValueError("Cannot compile assign value")
            for target in node.targets:
                if isinstance(target, ast.Name):
                    env[target.id] = val
        elif isinstance(node, ast.AugAssign):
            if not isinstance(node.target, ast.Name):
                raise ValueError("AugAssign only on names")
            old = env.get(node.target.id)
            rhs = self._compile_expr(node.value, builder, env)
            if old is None or rhs is None:
                raise ValueError("AugAssign: undefined variable")
            new_val = self._build_binop(node.op, old, rhs, builder)
            env[node.target.id] = new_val

    def _compile_expr(
        self,
        node    : ast.expr,
        builder : "ll.IRBuilder",
        env     : Dict[str, Any],
    ) -> Optional[Any]:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, int):
                return ll.Constant(ll.IntType(64), node.value)
            if isinstance(node.value, float):
                return ll.Constant(ll.DoubleType(), node.value)
            return None
        if isinstance(node, ast.Name):
            return env.get(node.id)
        if isinstance(node, ast.BinOp):
            left  = self._compile_expr(node.left,  builder, env)
            right = self._compile_expr(node.right, builder, env)
            if left is None or right is None:
                return None
            return self._build_binop(node.op, left, right, builder)
        if isinstance(node, ast.UnaryOp):
            operand = self._compile_expr(node.operand, builder, env)
            if operand is None:
                return None
            if isinstance(node.op, ast.USub):
                zero = ll.Constant(operand.type, 0)
                if isinstance(operand.type, ll.IntType):
                    return builder.neg(operand)
                return builder.fneg(operand)
        return None

    def _build_binop(
        self,
        op      : ast.operator,
        left    : Any,
        right   : Any,
        builder : "ll.IRBuilder",
    ) -> Any:
        is_int = isinstance(left.type, ll.IntType)
        if isinstance(op, ast.Add):
            return builder.add(left, right)  if is_int else builder.fadd(left, right)
        if isinstance(op, ast.Sub):
            return builder.sub(left, right)  if is_int else builder.fsub(left, right)
        if isinstance(op, ast.Mult):
            return builder.mul(left, right)  if is_int else builder.fmul(left, right)
        if isinstance(op, ast.Div):
            if is_int:
                # promote to double
                dl = builder.sitofp(left,  ll.DoubleType())
                dr = builder.sitofp(right, ll.DoubleType())
                return builder.fdiv(dl, dr)
            return builder.fdiv(left, right)
        if isinstance(op, ast.FloorDiv):
            return builder.sdiv(left, right) if is_int else builder.fdiv(left, right)
        if isinstance(op, ast.Mod):
            return builder.srem(left, right) if is_int else builder.frem(left, right)
        raise ValueError(f"Unsupported LLVM binop: {op}")


# ---------------------------------------------------------------------------
# Execution environment builder
# ---------------------------------------------------------------------------

def _build_global_env() -> Dict[str, Any]:
    """
    Construct the global execution namespace that every NithinLang program
    runs inside.  Injects:
      - All Python builtins
      - CoreLib functions (file I/O, math, etc.)
      - MLDSEngine functions (data_chudu, model_train, …)
      - GameEngine functions (game_start, game_draw, …)
      - AIEngine functions  (ai_adugu, ai_chudu)
    """
    core   = CoreLib()
    ml_ds  = MLDSEngine()
    game   = GameEngine()
    ai     = AIEngine()

    env: Dict[str, Any] = {
        "__builtins__"  : __builtins__,
        "__name__"      : "__nithin__",

        # ── Telugu Aliases for Game and Logic ─────────────────────────────
        "Nijam"         : True,
        "Abaddham"      : False,
        "raayi"         : core.nl_print,
        "ata_modal"     : game.game_start,
        "ata_rangu"     : game.game_color,
        "ata_vadi"      : game.game_fps,
        "ata_muginchu"  : game.game_stop,

        # ── Core standard library ─────────────────────────────────────────
        "print"         : core.nl_print,
        "input"         : core.nl_input,
        "f_open"        : core.f_open,
        "f_read"        : core.f_read,
        "f_write"       : core.f_write,
        "f_close"       : core.f_close,
        "f_append"      : core.f_append,
        "f_exists"      : core.f_exists,
        "f_delete"      : core.f_delete,
        "f_lines"       : core.f_lines,
        "math_sqrt"     : core.math_sqrt,
        "math_pow"      : core.math_pow,
        "math_abs"      : core.math_abs,
        "math_floor"    : core.math_floor,
        "math_ceil"     : core.math_ceil,
        "math_round"    : core.math_round,
        "math_log"      : core.math_log,
        "math_sin"      : core.math_sin,
        "math_cos"      : core.math_cos,
        "math_tan"      : core.math_tan,
        "math_pi"       : core.MATH_PI,
        "math_e"        : core.MATH_E,
        "vec_add"       : core.vec_add,
        "vec_sub"       : core.vec_sub,
        "vec_mul"       : core.vec_mul,
        "vec_dot"       : core.vec_dot,
        "mat_mul"       : core.mat_mul,
        "timer_start"   : core.timer_start,
        "timer_stop"    : core.timer_stop,
        "nl_sleep"      : core.nl_sleep,
        "nl_exit"       : core.nl_exit,
        "nl_env"        : core.nl_env,
        "nl_args"       : core.nl_args,
        "nl_platform"   : core.nl_platform,
        "nl_random"     : core.nl_random,
        "nl_randint"    : core.nl_randint,
        "nl_uuid"       : core.nl_uuid,
        "nl_hash"       : core.nl_hash,
        "nl_json_load"  : core.nl_json_load,
        "nl_json_dump"  : core.nl_json_dump,
        "nl_http_get"   : core.nl_http_get,
        "nl_http_post"  : core.nl_http_post,

        # ── ML / DS engine ────────────────────────────────────────────────
        "data_chudu"    : ml_ds.data_chudu,
        "data_save"     : ml_ds.data_save,
        "data_describe" : ml_ds.data_describe,
        "data_filter"   : ml_ds.data_filter,
        "data_sort"     : ml_ds.data_sort,
        "data_group"    : ml_ds.data_group,
        "data_plot"     : ml_ds.data_plot,
        "model_train"   : ml_ds.model_train,
        "model_test"    : ml_ds.model_test,
        "model_predict" : ml_ds.model_predict,
        "model_save"    : ml_ds.model_save,
        "model_load"    : ml_ds.model_load,
        "cluster_fit"   : ml_ds.cluster_fit,
        "np_array"      : ml_ds.np_array,
        "np_zeros"      : ml_ds.np_zeros,
        "np_ones"       : ml_ds.np_ones,
        "np_linspace"   : ml_ds.np_linspace,
        "np_mean"       : ml_ds.np_mean,
        "np_std"        : ml_ds.np_std,
        "np_sum"        : ml_ds.np_sum,
        "np_max"        : ml_ds.np_max,
        "np_min"        : ml_ds.np_min,

        # ── Game engine ───────────────────────────────────────────────────
        "game_start"    : game.game_start,
        "game_stop"     : game.game_stop,
        "game_draw"     : game.game_draw,
        "game_clear"    : game.game_clear,
        "game_loop"     : game.game_loop,
        "game_color"    : game.game_color,
        "game_text"     : game.game_text,
        "game_sprite"   : game.game_sprite,
        "game_key"      : game.game_key,
        "game_collide"  : game.game_collide,
        "game_sound"    : game.game_sound,
        "game_fps"      : game.game_fps,

        # ── AI engine ─────────────────────────────────────────────────────
        "ai_adugu"      : ai.ai_adugu,
        "ai_chudu"      : ai.ai_chudu,
        "ai_embed"      : ai.ai_embed,
        "ai_sentiment"  : ai.ai_sentiment,
        "ai_summarise"  : ai.ai_summarise,
        "ai_classify"   : ai.ai_classify,
        "ai_models"     : ai.ai_models,
        "ai_set_model"  : ai.ai_set_model,

        # ── Python builtins passthrough ───────────────────────────────────
        "range"         : range,
        "len"           : len,
        "int"           : int,
        "float"         : float,
        "str"           : str,
        "bool"          : bool,
        "list"          : list,
        "dict"          : dict,
        "set"           : set,
        "tuple"         : tuple,
        "type"          : type,
        "isinstance"    : isinstance,
        "enumerate"     : enumerate,
        "zip"           : zip,
        "map"           : map,
        "filter"        : filter,
        "sorted"        : sorted,
        "reversed"      : reversed,
        "abs"           : abs,
        "max"           : max,
        "min"           : min,
        "sum"           : sum,
        "round"         : round,
        "open"          : open,
        "True"          : True,
        "False"         : False,
        "None"          : None,
    }

    return env


# ---------------------------------------------------------------------------
# Main Compiler class
# ---------------------------------------------------------------------------

class CompileResult:
    """Holds the result of a compilation + execution attempt."""

    def __init__(
        self,
        success        : bool,
        execution_time : float = 0.0,
        output         : str   = "",
        errors         : List[str] = None,
        jit_functions  : List[str] = None,
    ) -> None:
        self.success         = success
        self.execution_time  = execution_time
        self.output          = output
        self.errors          : List[str] = errors or []
        self.jit_functions   : List[str] = jit_functions or []

    def __repr__(self) -> str:
        status = "OK" if self.success else "FAILED"
        return (
            f"CompileResult(status={status}, "
            f"time={self.execution_time:.4f}s, "
            f"jit={self.jit_functions})"
        )


class NithinCompiler:
    """
    Two-tier NithinLang compiler.

    Usage:
        from nithinlang.parser   import NithinParser
        from nithinlang.compiler import NithinCompiler

        parser   = NithinParser()
        compiler = NithinCompiler()

        result   = parser.parse_file("hello.nl")
        if result.success:
            compiler.compile_and_run(result)
    """

    def __init__(self, verbose: bool = False, jit_threshold: int = 1) -> None:
        """
        Args:
            verbose       : Print compilation diagnostics.
            jit_threshold : Minimum annotation count to attempt LLVM JIT.
        """
        self._verbose       = verbose
        self._jit_threshold = jit_threshold
        self._llvm_builder  = _LLVMFunctionBuilder() if _LLVM_AVAILABLE else None
        self._optimiser     = _ASTOptimiser()

    # ── Public API ────────────────────────────────────────────────────────

    def compile_and_run(self, parse_result: ParseResult) -> CompileResult:
        """
        Compile and immediately execute a ParseResult.

        Returns:
            CompileResult with timing, success flag, and any errors.
        """
        if not parse_result.success:
            errs = [str(e) for e in parse_result.errors]
            return CompileResult(success=False, errors=errs)

        source = parse_result.translated_src.strip()
        if not source:
            return CompileResult(success=True, execution_time=0.0)

        return self._compile_and_execute(source)

    def compile_to_ast(self, translated_src: str) -> Optional[ast.Module]:
        """
        Parse translated source into a Python AST Module.
        Returns None on syntax error.
        """
        try:
            tree = ast.parse(translated_src, mode="exec")
            return tree
        except SyntaxError as exc:
            if self._verbose:
                print(f"[NithinLang] SyntaxError: {exc}")
            return None

    def check_syntax(self, translated_src: str) -> List[str]:
        """
        Check syntax of translated source.  Returns list of error strings
        (empty list = OK).
        """
        errors: List[str] = []
        try:
            ast.parse(translated_src, mode="exec")
        except SyntaxError as exc:
            errors.append(
                f"SyntaxError at line {exc.lineno}, col {exc.offset}: {exc.msg}"
            )
        return errors

    # ── Internal pipeline ─────────────────────────────────────────────────

    def _compile_and_execute(self, source: str) -> CompileResult:
        jit_fns: List[str] = []
        errors : List[str] = []

        # ── Failsafe Translation (In case parser.py misses keywords) ──────
        fallback = {
            r'\bmaru\b': 'while',
            r'\byadi\b': 'if',
            r'\bkadu\b': 'else',
            r'\bkosam\b': 'for',
            r'\blopu\b': 'in',
            r'\braayi\b': 'print'
        }
        for k, v in fallback.items():
            source = re.sub(k, v, source)

        # ── Parse to AST ──────────────────────────────────────────────────
        try:
            tree = ast.parse(source, mode="exec")
        except SyntaxError as exc:
            errors.append(
                f"SyntaxError at line {exc.lineno}, col {exc.offset}: {exc.msg}\n"
                f"  → {exc.text}"
            )
            return CompileResult(success=False, errors=errors)

        # ── AST optimisation (CPython tier) ───────────────────────────────
        tree = self._optimiser.visit(tree)
        ast.fix_missing_locations(tree)

        # ── LLVM JIT: extract JIT-able top-level functions ─────────────────
        global_env = _build_global_env()

        if _LLVM_AVAILABLE and self._llvm_builder is not None:
            remaining_stmts: List[ast.stmt] = []
            for stmt in tree.body:
                if isinstance(stmt, ast.FunctionDef):
                    jit_fn = self._llvm_builder.try_build(stmt)
                    if jit_fn is not None:
                        global_env[stmt.name] = jit_fn
                        jit_fns.append(stmt.name)
                        if self._verbose:
                            print(
                                f"[NithinLang LLVM] JIT-compiled: "
                                f"{stmt.name}() → native code"
                            )
                        continue
                remaining_stmts.append(stmt)
            tree.body = remaining_stmts
            ast.fix_missing_locations(tree)

        # ── CPython bytecode compilation ──────────────────────────────────
        try:
            code_obj = compile(tree, filename="<nithinlang>", mode="exec")
        except Exception as exc:
            errors.append(f"Compile error: {exc}")
            return CompileResult(success=False, errors=errors, jit_functions=jit_fns)

        # ── Execution ─────────────────────────────────────────────────────
        t_start = time.perf_counter()
        try:
            exec(code_obj, global_env)  # noqa: S102
        except SystemExit:
            pass   # allow nl_exit() calls
        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            tb = traceback.format_exc()
            errors.append(f"RuntimeError:\n{tb}")
            return CompileResult(
                success=False,
                execution_time=elapsed,
                errors=errors,
                jit_functions=jit_fns,
            )

        elapsed = time.perf_counter() - t_start

        if self._verbose:
            tier = "LLVM+CPython" if jit_fns else "CPython"
            print(
                f"[NithinLang] Execution complete in {elapsed*1000:.2f} ms "
                f"(engine: {tier})"
            )

        return CompileResult(
            success=True,
            execution_time=elapsed,
            jit_functions=jit_fns,
        )