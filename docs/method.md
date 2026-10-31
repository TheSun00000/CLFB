# Method

Status: skeleton written from the code (`fb_continual.py`). To be completed and checked against the chat project and the paper draft.

## Problem
A pretrained FB model gives a forward embedding `F(s,a,z)` and a backward embedding `B(s)`. A reward `r` is encoded as `z_r = E_s[r(s) B(s)]` (up to normalization). Tasks `r_1, ..., r_K` arrive one at a time. Goal: keep encoding new rewards in the same model at fixed rank `d`, while the latents of earlier tasks stay valid after the model changes.

## Notation
- `θ_k`: model after stage k. `θ_old = θ_(k-1)`, frozen during stage k.
- `d`: latent dimension (`z_dim`).
- `Σ_θ = E[B_θ B_θᵀ]` over dataset states (estimated with `n_est` states per step, `n_big` states per stage).
- `A_θ`: cross-covariance between `B_θ` and `B_old`. `T = A Σ_old⁻¹`: map that carries an old latent into the new basis.
- `r̄_k`: normalized reward of task k. `D_k`: labeled set (`n_labeled` states).
- `ν`: distribution over latents, `(1-α)` uniform sphere + `α` uniform mixture of stored atoms `z̄_j`, j < k.

## Stage k objective
`L_k = L_FB + λ L_rec(r_k) + μ_B R^B + μ_Q R^Q`

- `L_FB`: the usual FB loss, classical variant (actor in the bootstrap), with a DDPG actor (optional TD3+BC weight `bc_coef`).
- `L_rec`: detached-coefficient regression of `r̄_k` on `B_θ` over `D_k`. Optional cross-fitting (`z_hat` on one half, loss on the other).
- `R^B`: `E_(z~ν)[ zᵀ M z / zᵀ Σ_old⁻¹ z ]` with `M = Σ_old⁻¹ − Tᵀ Σ_θ⁻¹ T`.
- `R^Q`: value matching between old and new model on the re-encoded old family, `z' = sg(T z)`. Both heads of F are matched head to head, `(F_θ^i(s,a,z̄')ᵀ z' − F_old^i(s,a,z)ᵀ z)²`, averaged over i = 1, 2.

## End of stage
- `z_j ← T z_j` for j < k, with `T` between `θ_k` and `θ_(k-1)` on a fresh batch of `n_big` states.
- `z_k = mean_i r̄_k(s_i) B_θk(s_i)`.
- Only d numbers per task are stored.

## Evaluation
- fresh: z inferred from fresh 10k labels, zero-shot, as in the pretraining script.
- tracked: z = stored latent carried through the T's. Uses the scalar `m2_j` (second moment of `r_j` on `D_j`) to put the reward on the scale of `r̄_j`. Evaluation only, never used in training.
- `eps`: fraction of the reward energy not explained by B on held-out states.
- `z_shrink`: mean ‖T z‖ / ‖z‖.

## Open points
- (to fill) why this form of `R^B`, and how it relates to the retention bound.
- (to fill) relation to the earlier "Implicit FB / SF" work.
- (to fill) baselines to compare against.
