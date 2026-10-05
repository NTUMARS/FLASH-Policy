# MIP: Minimal Iterative Policy (baseline)

Ported from the official implementation of **"Much Ado About Noising: Dispelling
the Myths of Generative Robotic Control"** (arXiv 2512.01809), specifically the
**simplified form** which is the official default:

- Training loss: `mip/losses.py::mip_loss`
- Inference sampler: `mip/samplers.py::mip_sampler`

(The official code comments call it "Minimum iterative policy"; the paper and
common usage say "Minimal Iterative Policy". We follow the paper.)

## Algorithm

All in the normalized action space (LinearNormalizer, limits mode -> [-1, 1],
matching the official MinMax normalization). The network directly predicts the
action (not a velocity field), conditioned on a continuous time input.
`t* = t_two_step = 0.9` is fixed.

**Training** (two supervised terms, both regressing the GT action `a`):

```
pred_0 = net(x = 0,               t = 0,  obs)
pred_1 = net(x = a + (1-t*)·z,    t = t*, obs),   z ~ N(0, I)
loss   = loss_scale · mean( ‖pred_0 − a‖²/t*² + ‖pred_1 − a‖²/(1−t*)² )
```

where `‖·‖²` is squared L2 summed over the action dim (official `get_norm`).

**Inference** (fully deterministic, obs encoded once):

```
â₀ = net(0,  t = 0,  obs)
â  = net(â₀, t = t*, obs)
```

## Fair comparison

Everything except the MIP objective is kept **identical to fm_dit**: the
ResNet18 `MultiImageObsEncoder`, the `FlowTransformer` DiT backbone
(hidden 512 / 4 layers / 8 heads / dropout 0.1), `horizon` / `n_obs_steps` /
`n_action_steps`, the normalizer, and all training hyperparameters
(train_config/default_train.yaml). MIP-specific parameters (`t_two_step=0.9`,
`loss_scale=0.1`, `norm_type=l2`) follow the official experiment config
(`examples/configs/optimization/default.yaml`).

## Usage

```bash
bash roboverse_learn/il/il_run.sh --task_name_set stack_cube --policy_name mip ...
```
