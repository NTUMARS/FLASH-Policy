# RoboVerse Imitation Learning (IL) Policies

## Example Usage

Create and activate your own Python environment using the repository's [environment configuration guide](../../ENVIRONMENT.md) before running conversion, training, evaluation, or tests. Install the simulator and shared policy dependencies from the root `requirements.txt` in that environment.

Pick a policy folder and follow its README for setup and usage. The [FLASH paper protocol](../../README.md#demonstrations-and-training) specifies the demonstration counts and settings for the five simulation tasks: 99 Close Box demonstrations, 100 each for Pick Cube and Stack Cube, and 40 each for the two LIBERO tasks. Unless stated otherwise, all policies are configured for 15,000 training steps with checkpoints every 1,250 steps; Table 1 evaluates the 10,000-step checkpoint.

For all five tasks, follow the [simulation reproduction guide](../../REPRODUCING.md), which includes demonstration collection, conversion, FLASH training, and evaluation. The paper's **Open Drawer** is `libero_90.kitchen_scene1_open_bottom_drawer` (the bottom drawer in kitchen scene 1), and **Pick-Place Bowl** is `libero_90.kitchen_scene1_put_the_black_bowl_on_the_plate` (the black bowl on the plate in kitchen scene 1). Both use `--sim_set mujoco`; use these full identifiers with `--task_name_set`.

`il_run.sh` applies these task defaults when no training budget is supplied. Explicit `--num_steps`, `--num_epochs`, `--demo_num`, and `--eval_ckpt_name` override the corresponding defaults. The 2D experiments retain their separate protocol.

Example:

```bash
# From the repository root, with your configured environment active.
python -m pip install -r requirements.txt

# Train a baseline; choose ddpm_dit, vita, or fm_dit as appropriate.
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name ddpm_dit \
    --demo_num 100 --num_steps 15000 --checkpoint_every_n_steps 1250 \
    --train_enable True --eval_enable False

# Evaluate the Table 1 checkpoint from that training run.
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name ddpm_dit \
    --demo_num 100 --num_steps 15000 --eval_ckpt_name 10000 \
    --train_enable False --eval_enable True
```

We keep each policy as self-contained as possible (code, dependencies, docs) and only share the minimum common abstractions.

## Custom 2D tasks

[CorridorPush and ForkReach](push2d/README.md) include standalone environments,
paired expert demonstrations, training routes and evaluation tools. The published
task names are `corridor_push_lv2` and `switchshot_N_lv4`. The 2D route supports
FLASH, FM-DiT, ACT and the other configured image policies.

## Troubleshooting

```bash
# Fix potential package version issues
bash roboverse_learn/il/il_setup.sh
```

## Supported Algorithms

| Name | Policy | Backbone | Model Config | Ref |
| --- | --- | --- | --- | --- |
| `ddpm_dit` | Diffusion Policy (DDPM) | DiT | `model_config/ddpm_dit.yaml` | [1], [5] |
| `fm_dit` | Flow Matching | DiT | `model_config/fm_dit.yaml` | [6], [5] |
| `vita` | VITA Policy | MLP | `model_config/vita.yaml` | [7] |
| `ddpm_unet` | Diffusion Policy (DDPM) | UNet | `model_config/ddpm.yaml` | [1], [4] |
| `ddim_unet` | Diffusion Policy (DDIM) | UNet | `model_config/ddim.yaml` | [2], [4] |
| `fm_unet` | Flow Matching | UNet | `model_config/fm_unet.yaml` | [6] |
| `score_unet` | Score-Based Model | UNet | `model_config/score.yaml` | [3], [4] |

**References**

1. Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising Diffusion Probabilistic Models." (2020).  
2. Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising Diffusion Implicit Models." (2021).  
3. Song, Yang, et al. "Score-Based Generative Modeling through Stochastic Differential Equations." (2021).  
4. Chi, Cheng, et al. "Diffusion Policy: Diffusion Models for Robotic Manipulation." (2023).  
5. Peebles, William, and Jun-Yan Zhu. "DiT: Diffusion Models with Transformers." (2023).  
6. Lipman, Yaron, et al. "Flow Matching for Generative Modeling." (2023).  
7. Gao, Dechen, et al. "VITA: Vision-to-Action Flow Matching Policy." (2025).

### Single-step Generation

We also include MeanFlow [1, 2], Improved MeanFlow (iMF) [3], and Consistency Flow Matching (CFM) [4] for FM generation.

**References**

[1] Geng, Zhengyang, et al. "Mean flows for one-step generative modeling." arXiv preprint arXiv:2505.13447 (2025).
[2] Sheng, Juyi, et al. "MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation." arXiv preprint arXiv:2507.10543 (2025).
[3] Geng, Zhengyang, et al. "Improved Mean Flows: On the Challenges of Fastforward Generative Models." arXiv preprint arXiv:2512.02012 (2025).
[4] Yang, Ling, et al. "Consistency flow matching: Defining straight flows with velocity consistency." arXiv preprint arXiv:2407.02398 (2024).
