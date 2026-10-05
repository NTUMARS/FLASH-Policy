"""Shared ``step{n}`` naming segment for eval artifacts.

Single source of truth for the inference-step count that goes into eval
directory / summary-file names, so metasim (``DefaultRunner``), native LIBERO
(``eval_libero``) and push2d all spell it the same way. Sibling of
``recovery_naming.py``, which plays the same role for the recovery knobs.

WHY A RESOLVER AND NOT A BARE ``getattr``: the same quantity — "how many solver
steps does one inference take" — is spelled differently across policy families,
and the split is historical, not semantic:

  * ``num_inference_steps`` : fm_dit / fm_unet / ddpm_* / ddim_unet / flash /
    flash_g / apf / score / a2a_mini.  ACT exposes it as a read-only property
    fixed at 1 (CVAE decodes in one shot), so it also lands here truthfully.
  * ``num_sampling_steps``  : a2a / a2a_noise / vita (read off their
    ``flow_matcher``).

Reading only the first name — as ``default_runner.py`` historically did — makes
every a2a/vita run fall back to the "unknown" tag even though the value is right
there under the other name.
"""
from __future__ import annotations

# Checked in order; first attribute that exists AND is not None wins.
_STEP_ATTRS = ("num_inference_steps", "num_sampling_steps")

UNKNOWN_TAG = "stepNA"


def steps_attr_name(policy) -> str | None:
    """Which of the two spellings this policy actually carries, or ``None``.

    Reporting uses this so ``00_eval_config.txt`` prints the TRUE attribute name
    (project rule eval-config-txt-params) instead of silently renaming an a2a
    policy's ``num_sampling_steps`` into ``num_inference_steps``.
    """
    if policy is None:
        return None
    for attr in _STEP_ATTRS:
        if getattr(policy, attr, None) is not None:
            return attr
    return None


def inference_steps(policy) -> int | None:
    """The resolved per-inference solver-step count, or ``None`` if the policy
    exposes neither spelling (e.g. the LIBERO ``expert`` replay path, which has
    no policy object at all — pass ``None`` and get ``None`` back)."""
    if policy is None:
        return None
    for attr in _STEP_ATTRS:
        val = getattr(policy, attr, None)
        if val is not None:
            return int(val)
    return None


def step_tag(policy) -> str:
    """Naming segment: ``step10`` / ``step1`` / ``stepNA``.

    ``stepNA`` (rather than omitting the segment) keeps every name the same
    shape, so downstream globs and split-on-``_`` positions stay stable — same
    convention as ``push2d_eval.rename_with_metrics``.
    """
    n = inference_steps(policy)
    return UNKNOWN_TAG if n is None else f"step{n}"
