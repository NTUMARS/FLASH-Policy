"""Fail fast on unknown policy constructor arguments.

Why this is needed
------------------
A misspelled ``--override`` can otherwise allow an experiment to run while the
parameter has no effect. Hours later, two intended variants may turn out to be
identical and the resulting batch unusable.

Hydra's struct mode covers only part of this problem: it rejects keys absent
from the YAML (``Could not override 'x'. To append use +x=...``), but it cannot
catch the following cases:

1. **YAML/constructor drift**: a key exists in the policy YAML but is not a
   named constructor argument, for example after adding a YAML parameter but
   forgetting to update ``__init__``. Hydra applies and passes the override,
   which then falls through ``**kwargs`` -> ``self.kwargs`` ->
   ``conditional_sample(**self.kwargs)`` and is ignored.
2. **The `+key=value` append syntax**: this explicitly bypasses struct checking
   and can disappear in the same way.
3. **Obsolete keys in old checkpoints**: evaluation rebuilds a policy from the
   checkpoint's saved config. A removed option such as
   ``progress_now_weight`` could otherwise disappear silently, producing a
   network different from the one the evaluator expects.

The dp family (including ``DiffusionDenoisingImagePolicy``) has no ``**kwargs``,
so Python already raises ``TypeError``. This module applies the same standard to
the remaining policies.

Usage
-----
Call this function at the start of ``__init__`` so invalid arguments fail before
an expensive model is built::

    def __init__(self, shape_meta, ..., **kwargs):
        reject_unknown_kwargs(self, kwargs)
        super().__init__()

If an option must be forwarded downstream, such as DDIM passing ``eta`` to
``scheduler.step``, register it explicitly with ``allowed`` instead of disabling
the guard::

    reject_unknown_kwargs(self, kwargs, allowed=("eta", "use_clipped_model_output"))
"""

from __future__ import annotations

import difflib
import inspect
from typing import Iterable

__all__ = ["reject_unknown_kwargs", "named_init_params"]


def named_init_params(cls: type) -> set[str]:
    """Return named ``__init__`` parameters for ``cls`` and all base classes.

    ``**kwargs`` itself is excluded. The result supports "did you mean" hints
    for misspelled keys.
    """
    out: set[str] = set()
    for c in getattr(cls, "__mro__", (cls,)):
        init = c.__dict__.get("__init__")
        if init is None:
            continue
        try:
            sig = inspect.signature(init)
        except (ValueError, TypeError):  # C implementations such as object.__init__
            continue
        out |= {
            n for n, p in sig.parameters.items()
            if n != "self" and p.kind is not inspect.Parameter.VAR_KEYWORD
        }
    return out


def reject_unknown_kwargs(
    owner: object,
    kwargs: dict,
    *,
    allowed: Iterable[str] = (),
) -> None:
    """Raise ``TypeError`` immediately for unknown constructor arguments.

    Args:
        owner: Policy instance. Its concrete class name is used in errors so a
            bad subclass configuration names the subclass rather than the base
            class that implements the guard.
        kwargs: ``**kwargs`` received by ``__init__``.
        allowed: Names explicitly permitted because downstream code consumes
            them.

    Raises:
        TypeError: If ``kwargs`` contains any key outside ``allowed``.
    """
    allowed = set(allowed)
    unknown = sorted(k for k in kwargs if k not in allowed)
    if not unknown:
        return

    cls = type(owner)
    known = named_init_params(cls) | allowed
    lines = [
        f"{cls.__name__} received {len(unknown)} unrecognized constructor "
        f"argument(s): {unknown}.",
        "",
        "These keys are neither named constructor parameters nor explicitly",
        "allowed passthrough options. Letting them through would silently ignore",
        "them while the experiment continues, potentially making two variants",
        "identical. Construction has therefore been stopped.",
        "",
        "Common causes:",
        "  1. A misspelled --override policy_config.<key>. Hydra's `+key=value`",
        "     append syntax bypasses its struct check.",
        "  2. A parameter was added to the policy YAML but not to __init__.",
        "  3. The evaluated checkpoint was trained with older code and its saved",
        "     config contains a parameter that has since been removed.",
    ]
    for k in unknown:
        near = difflib.get_close_matches(k, known, n=3, cutoff=0.6)
        if near:
            lines.append(f"  - {k!r} -> did you mean: {', '.join(repr(n) for n in near)}?")
    lines += [
        "",
        "Run `python scripts/check_experiments.py` before a batch to detect these",
        "errors statically.",
    ]
    raise TypeError("\n".join(lines))
