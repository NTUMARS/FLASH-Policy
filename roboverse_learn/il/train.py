import os
import pathlib
import sys

import hydra
from omegaconf import OmegaConf, open_dict
from loguru import logger as log

here = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(here)
sys.path.insert(0, project_root)
from roboverse_learn.il.runners.base_runner import BaseRunner

abs_config_path = str(pathlib.Path(__file__).resolve().parent.joinpath("configs").absolute())
OmegaConf.register_new_resolver("eval", eval, replace=True)


@hydra.main(config_path=abs_config_path, version_base="1.3")
def main(cfg):
    # ===== Proprioceptive ablation switch =====
    # When use_proprioception=False, Remove the agent_pos from the observations and retain only the image condition
    # must delete shape_meta.obs.agent_pos before OmegaConf.resolve so that all interpolated references to
    # shape_meta (obs_encoder / dataset / policy) lose agent_pos synchronously.
    # Default is True, which matches the baseline exactly.
    if not cfg.get("use_proprioception", True):
        _obs_meta = cfg.shape_meta.obs
        if "agent_pos" in _obs_meta:
            with open_dict(_obs_meta):
                del _obs_meta["agent_pos"]
            log.info("[use_proprioception=False] Removed agent_pos from shape_meta.obs -> image-only condition.")
    OmegaConf.resolve(cfg)

    cls = hydra.utils.get_class(cfg._target_)

    # Isaac Sim leaves background threads that prevent normal interpreter
    # shutdown, causing the process to hang after training/eval completes
    # (or crashes mid-way). Wrap everything in try/finally so os._exit is
    # guaranteed to run regardless of success, failure, or hang.
    exit_code = 0
    try:
        runner: BaseRunner = cls(cfg)
        runner.run()
        log.info("All tasks finished successfully.")
    except Exception as e:
        log.error(f"Run failed with error: {e}")
        exit_code = 1
    finally:
        log.info("Forcing process exit to release GPU memory.")
        sys.stdout.flush()
        sys.stderr.flush()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass
        os._exit(exit_code)


if __name__ == "__main__":
    main()
