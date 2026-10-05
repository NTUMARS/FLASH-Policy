"""Build the ``eval_config.txt`` body for an evaluation run.

This module holds the *pure* formatting logic (no torch / hydra / metasim
imports) so it can be unit-tested in isolation and reused across runners.

Rule (project memory ``eval-config-txt-params``): when a policy's yaml gains a
new parameter, that parameter's value for THIS run must be recorded here -- and
parameters specific to *other* policy families must NOT leak into an unrelated
policy's report. We therefore gate each family-specific block behind a
``hasattr`` check on the live policy object instead of printing every key
unconditionally.
"""

from __future__ import annotations

from roboverse_learn.il.utils.step_naming import steps_attr_name


def _kv(key, value, suffix=""):
    """Format one ``key = value`` line with the ``=`` column-aligned.

    Width 26 keeps the longest current key (``recovery_cooldown_chunks``, 24
    chars) aligned; bump it if a longer key is added later.
    """
    line = f"  {key:<26}= {value}"
    return f"{line}{suffix}"


def chunk_lines(policy_obj):
    """Report the fm_dit_chunk family using ``chunk_extra_steps`` as its marker.

    These training-time architecture parameters (canvas length and discarded
    tail) come from the checkpoint config during evaluation. Recording them
    makes runs with different canvas configurations distinguishable in
    eval_config.txt.
    """
    if not hasattr(policy_obj, "chunk_extra_steps"):
        return []
    lines = ["  --- Chunk Canvas (future-only) ---"]
    lines.append(_kv("chunk_extra_steps", policy_obj.chunk_extra_steps))
    lines.append(_kv("chunk_len", getattr(policy_obj, "chunk_len", "N/A")))
    return lines


def hist_start_lines(policy_obj):
    """Report the hist-start family using the fm_dit-only ``hist_start_s`` marker.

    LIBERO's 00_eval_config.txt does not use ``build_eval_config_lines``;
    eval_libero.py writes its own [LiberoEvalArgs] block. Its ``key: value``
    format can coexist with the aligned ``_kv`` format here, so both paths call
    this function directly as the single source of truth. Calibration artifacts
    are excluded from Git under data_policy; gains and an MD5 are recorded here
    for provenance.
    """
    if not hasattr(policy_obj, "hist_start_s"):
        return []
    lines = ["  --- Hist-Start Inference ---"]
    s_val = getattr(policy_obj, "hist_start_s", 0.0)
    lines.append(_kv("hist_start_s", s_val))
    lines.append(_kv("hist_start_layout", getattr(policy_obj, "hist_start_layout", "replay")))
    lines.append(_kv("hist_start_mode", getattr(policy_obj, "hist_start_mode", "delta_ee")))
    lines.append(_kv("hist_start_grip", getattr(policy_obj, "hist_start_grip", "minmax")))
    lines.append(_kv("hist_start_noise_std", getattr(policy_obj, "hist_start_noise_std", 0.0)))
    lines.append(_kv("hist_start_noise_dim_mask", getattr(policy_obj, "hist_start_noise_dim_mask", None)))
    lines.append(_kv("flow_start_noise_seed", getattr(policy_obj, "flow_start_noise_seed", None)))
    # When per-episode mode is enabled, state explicitly that z0 does not consume
    # the isolated stream. Otherwise, alongside episode_rng_isolation_effective,
    # readers may assume that stream selects z0; each episode actually rebuilds
    # z0 from one fixed seed.
    _per_ep = bool(getattr(policy_obj, "flow_start_noise_per_episode", False))
    lines.append(_kv("flow_start_noise_per_episode", _per_ep,
                     suffix="  (z0 is fixed by the per-episode seed and does not consume the isolated episode stream)" if _per_ep else ""))
    calib = getattr(policy_obj, "hist_start_calib", None)
    lines.append(_kv("hist_start_calib", calib))
    mode = getattr(policy_obj, "hist_start_mode", "delta_ee")
    active = bool(s_val) and float(s_val) > 0.0
    if active and not calib and mode == "delta_ee":
        lines.append(_kv("hist_calib_gains", "REQUIRED but not set (inference will fail)"))
    elif active and calib:
        try:
            import hashlib

            from roboverse_learn.il.utils.hist_start import load_calib
            c = load_calib(calib)
            with open(calib, "rb") as f:
                md5 = hashlib.md5(f.read()).hexdigest()[:8]
            lines.append(_kv("hist_calib_gains", [round(g, 3) for g in c["gains"]]))
            lines.append(_kv("hist_calib_grip", c["grip"]))
            lines.append(_kv("hist_calib_md5", md5))
        except ValueError as e:
            lines.append(_kv("hist_calib_gains", f"MISSING/INVALID ({e})"))
    return lines


def _policy_family_blocks(policy_obj):
    """Per-family parameter blocks, each gated on a fingerprint attribute.

    Extracted verbatim out of build_eval_config_lines so native LIBERO can
    emit the same blocks; gating logic and emission order are unchanged.
    """
    lines = []
    # ---- FLASH / FLASH-G polynomial family (only when present) ----
    # ``basis_type`` exists on flash & flash_g but not on apf (which has a lone
    # ``poly_order``) nor on any other policy -> it cleanly gates this block.
    if hasattr(policy_obj, "basis_type"):
        lines.append("  --- FLASH Polynomial Parameters ---")
        lines.append(_kv("poly_order", getattr(policy_obj, "poly_order", "N/A")))
        lines.append(_kv("basis_type", policy_obj.basis_type))
        lines.append(_kv("fit_pad", getattr(policy_obj, "fit_pad", 0)))
        lines.append(_kv("boundary_constraint", getattr(policy_obj, "boundary_constraint", False)))
        lines.append(_kv("poly_prefix", getattr(policy_obj, "poly_prefix", None)))
        lines.append(_kv("poly_suffix", getattr(policy_obj, "poly_suffix", 0)))
        lines.append(_kv("H_poly", getattr(policy_obj, "_H_poly", "N/A")))

    # ---- FLASH solver params (flash only; flash_g has no flash_solver) ----
    if hasattr(policy_obj, "flash_solver"):
        lines.append("  --- FLASH Parameters ---")
        lines.append(_kv("flash_solver", policy_obj.flash_solver))
        lines.append(_kv("history_reg_lambda", policy_obj.history_reg_lambda, " (from checkpoint)"))
        lines.append(_kv("history_noise_std", policy_obj.history_noise_std, " (from checkpoint)"))
        # Adaptive-order block: gated on the flash_solver fingerprint above, NOT
        # on hasattr(use_adaptive_gate) — apf carries the same attribute name and
        # a bare hasattr would leak these lines into apf reports.  The two
        # always-shown keys make capped/uncapped and gated/ungated runs
        # distinguishable from eval_config.txt alone; effective_history_fit_order
        # is derived from the checkpoint buffer (stale-config proof).
        lines.append(_kv("history_fit_order",
                         getattr(policy_obj, "effective_history_fit_order",
                                 getattr(policy_obj, "poly_order", "N/A"))))
        lines.append(_kv("use_adaptive_gate", getattr(policy_obj, "use_adaptive_gate", False)))
        if getattr(policy_obj, "use_adaptive_gate", False):
            lines.append(_kv("gate_prior_high", policy_obj.gate_prior_high, " (from checkpoint)"))
            lines.append(_kv("gate_prior_low", policy_obj.gate_prior_low, " (from checkpoint)"))
            lines.append(_kv("sparse_weight", policy_obj.sparse_weight, " (from checkpoint)"))
            lines.append(_kv("gate_l1_start_order", policy_obj.gate_l1_start_order))
            lines.append(_kv("gate_recon_weight", policy_obj.gate_recon_weight))

    # ---- Progress-aware fm_dit family (only when present) ----
    if hasattr(policy_obj, "progress_curve_weight"):
        lines.append("  --- Progress-Aware Parameters ---")
        lines.append(_kv("progress_curve_weight", policy_obj.progress_curve_weight))
        lines.append(_kv("progress_head_hidden", getattr(policy_obj, "progress_head_hidden", "N/A")))
        lines.append(_kv("progress_head_image_only", getattr(policy_obj, "progress_head_image_only", False)))
        # Key-pruned progress variants, e.g. LIBERO imglang drops agent_pos but
        # keeps lang_emb. This is their only difference from the full-condition
        # variant and must be visible in the report. Old checkpoints lack this
        # attribute, so report an empty tuple, matching the image_only default.
        lines.append(_kv("progress_head_drop_keys",
                         tuple(getattr(policy_obj, "progress_head_drop_keys", ()) or ())))

    # ---- fm_dit_chunk family (only when present) ----
    lines.extend(chunk_lines(policy_obj))

    # ---- hist-start family (fm_dit only; gated by its fingerprint attr) ----
    lines.extend(hist_start_lines(policy_obj))

    # ---- ACT (CVAE) family (only when present) ----
    # ``kl_weight`` currently exists only on ACTImagePolicy. The legacy ACTPolicy
    # does not use this report, while a2a's YAML value is absorbed by **kwargs
    # without becoming an instance attribute. It is therefore an unambiguous
    # family marker. Every act.yaml policy parameter is recorded here.
    if hasattr(policy_obj, "kl_weight"):
        lines.append("  --- ACT (CVAE) Parameters ---")
        lines.append(_kv("chunk_size", getattr(policy_obj, "chunk_size", "N/A")))
        lines.append(_kv("kl_weight", policy_obj.kl_weight))
        lines.append(_kv("hidden_dim", getattr(policy_obj, "hidden_dim", "N/A")))
        lines.append(_kv("dim_feedforward", getattr(policy_obj, "dim_feedforward", "N/A")))
        lines.append(_kv("enc_layers", getattr(policy_obj, "enc_layers", "N/A")))
        lines.append(_kv("dec_layers", getattr(policy_obj, "dec_layers", "N/A")))
        lines.append(_kv("nheads", getattr(policy_obj, "nheads", "N/A")))
        lines.append(_kv("dropout", getattr(policy_obj, "dropout", "N/A")))
        lines.append(_kv("position_embedding", getattr(policy_obj, "position_embedding", "N/A")))
        lines.append(_kv("camera_names", getattr(policy_obj, "camera_names", "N/A")))
        lines.append(_kv("backbone", getattr(policy_obj, "backbone_name", "N/A")))
    return lines


def policy_family_lines(policy_obj):
    """Every block that is derived from the POLICY OBJECT alone — architecture
    plus the per-family parameter blocks.

    Split out of :func:`build_eval_config_lines` so native LIBERO
    (``eval_libero``, which writes its own ``[LiberoEvalArgs]`` header instead of
    the metasim control-side block) emits the SAME policy blocks as metasim and
    push2d. Before this split LIBERO only called ``chunk_lines`` +
    ``hist_start_lines`` and therefore silently omitted the architecture and
    Progress-Aware / FLASH / ACT blocks — the drift this function exists to make
    impossible.

    Contains nothing control-side and nothing suite-specific, so it is safe for
    any caller. Each family block is gated on a fingerprint attribute, so a
    policy only ever sees its own parameters (project rule
    eval-config-txt-params).
    """
    lines = []
    lines.append("  --- Model Architecture (from checkpoint) ---")
    lines.append(_kv("horizon", getattr(policy_obj, "horizon", "N/A")))
    lines.append(_kv("n_obs_steps", getattr(policy_obj, "n_obs_steps", "N/A")))
    lines.append(_kv("n_action_steps", getattr(policy_obj, "n_action_steps", "N/A")))
    # Solver-step count. The SAME quantity is spelled `num_inference_steps` by
    # the fm/dp/flash/act families and `num_sampling_steps` by a2a/a2a_noise/vita;
    # emit whichever the checkpoint actually carries (truthful key) — reading only
    # the first name used to print "N/A" for every a2a/vita run. The resolved
    # value is what the run dir's `step{n}` segment shows (utils/step_naming.py).
    _step_attr = steps_attr_name(policy_obj)
    if _step_attr is None:
        lines.append(_kv("num_inference_steps", "N/A"))
    else:
        lines.append(_kv(_step_attr, getattr(policy_obj, _step_attr)))
    if hasattr(policy_obj, "inference_mode"):
        lines.append(_kv("inference_mode", policy_obj.inference_mode))
    lines.extend(_policy_family_blocks(policy_obj))
    return lines


def build_eval_config_lines(policy_obj, eval_params, dr_level):
    """Return the list of text lines recording the parameters of one eval run.

    Args:
        policy_obj: the live policy instance being evaluated. Family-specific
            blocks are emitted only when the corresponding attributes exist on
            it (e.g. ``basis_type`` for the FLASH polynomial family,
            ``progress_curve_weight`` for the progress-aware fm_dit family).
        eval_params: dict of control-side eval parameters for this run.
        dr_level: domain-randomization level used for this run.

    NOTE: the joint-velocity PD knobs (``velocity_pd_gains`` /
    ``velocity_kp_scale`` / ``velocity_kd_scale``) and the per-joint Kp/Kd table
    are deliberately NOT reported: they are permanently left at their defaults,
    so printing them added noise to every report without ever distinguishing two
    runs. They remain live eval args (``utils/eval_args.py``) and are still
    applied by ``DefaultRunner``; only the reporting was dropped.
    """
    lines = []
    lines.append("=" * 70)
    lines.append("[Eval Config] Parameters used in THIS evaluation run:")

    # ---- Control-side params (every policy) ----
    lines.append(_kv("send_vel_target", eval_params["send_vel_target"]))
    lines.append(_kv("downsample_ratio", eval_params["downsample_ratio"]))
    lines.append(_kv("dr_level_eval", dr_level))
    lines.append(_kv("max_step", eval_params["max_step"]))

    # ---- Everything policy-derived (shared with native LIBERO) ----
    lines.extend(policy_family_lines(policy_obj))


    # ---- Recovery (retreat-then-replan, R1); control-side, gated by enabled ----
    rec = eval_params.get("recovery")
    if rec and (rec.get("enabled") or rec.get("report_when_disabled")):
        lines.append("  --- Recovery (R1) ---")
        lines.append(_kv("recovery_enable", rec["enabled"]))
        # push2d-only: the controller class name disambiguates the arm at a glance.
        # metasim/libero never set it -> .get() is None -> their reports are unchanged.
        if rec.get("controller") is not None:
            lines.append(_kv("recovery_controller", rec["controller"]))
        lines.append(_kv("recovery_w_retreat", rec["w_retreat"]))
        lines.append(_kv("recovery_retreat_steps", rec.get("retreat_steps", 0)))
        lines.append(_kv("recovery_retreat_interp", rec.get("retreat_interp", False)))
        lines.append(_kv("recovery_retreat_margin", rec.get("retreat_margin", 0)))
        lines.append(_kv("recovery_retreat_interp_steps", rec.get("retreat_interp_steps", -1)))
        lines.append(_kv("recovery_retreat_interp_max_dist", rec.get("retreat_interp_max_dist", -1.0)))
        lines.append(_kv("recovery_retreat_radius", rec.get("retreat_radius", 0.0)))
        lines.append(_kv("recovery_retreat_anchor", rec.get("retreat_anchor", False)))
        lines.append(_kv("recovery_retreat_gripper_replay", rec.get("retreat_gripper_replay", False)))
        lines.append(_kv("recovery_detector_w", rec.get("detector_w", rec.get("w_retreat"))))
        lines.append(_kv("recovery_stall_tau", rec["stall_tau"]))
        lines.append(_kv("recovery_stall_ceiling", rec["stall_ceiling"]))
        lines.append(_kv("recovery_stall_floor", rec["stall_floor"]))
        lines.append(_kv("recovery_cooldown_chunks", rec["cooldown_chunks"]))
        lines.append(_kv("recovery_r_max", rec["r_max"]))
        lines.append(_kv("recovery_settle", rec.get("settle", False)))
        lines.append(_kv("recovery_settle_steps", rec.get("settle_steps", -1)))
        lines.append(_kv("recovery_settle_joint_tol", rec.get("settle_joint_tol", 0.05)))
        lines.append(_kv("recovery_g0", rec.get("g0", 0.0)))
        lines.append(_kv("recovery_g0_chunks", rec.get("g0_chunks", 6)))
        lines.append(_kv("recovery_g0_ramp", rec.get("g0_ramp", 0.0)))
        lines.append(_kv("recovery_tau", rec.get("tau", 1.0)))
        lines.append(_kv("recovery_tau_ramp", rec.get("tau_ramp", 0.0)))
        # Task 10 (push2d-only sim upper bound): metasim/libero callers never
        # pass "oracle_rewind" in their recovery dict at all, so rec.get(...)
        # is None there and this line is skipped -- byte-identical regression
        # for those callers. push2d always passes it (True or False), so its
        # 00_eval_config.txt always records the flag explicitly.
        if rec.get("oracle_rewind") is not None:
            lines.append(_kv("recovery_oracle_rewind", rec["oracle_rewind"]))
        # push2d-only attribution arm (same gating rationale as oracle_rewind above:
        # metasim/libero never put the key in their recovery dict, so rec.get(...) is
        # None there and their 00_eval_config.txt stays byte-identical).  Without this
        # line the replan-only arm and the full-retreat arm produce IDENTICAL configs
        # — the two arms of the C/C' attribution split would be indistinguishable.
        if rec.get("replan_only") is not None:
            lines.append(_kv("recovery_replan_only", rec["replan_only"]))
            if rec["replan_only"]:
                # Be explicit that the retreat-shaping knobs above are dead on this arm
                # (ReplanOnlyController overrides _begin_retreat wholesale), so nobody
                # reads them as "what this run actually did".
                lines.append("  (replan_only=True: the recovery_retreat_* and "
                             "recovery_settle* settings above are inactive; "
                             "this variant replans in place when triggered)")

    lines.append("=" * 70)
    return lines
