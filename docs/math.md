# Math of the method

Full derivation of the method, step by step, in the notation of `fb_continual.py`. Formulas are LaTeX (`$...$`, `$$...$$`). The same text is the Math tab of the tracker; `docs/method.md` is the short summary. First version, 2026-11-02. The analysis (section 6) follows the LaTeX write-up of the method.

## Notation

- $\rho$: distribution of states in the offline dataset $D$; all expectations $\mathbb E_\rho$ are over it.
- $\gamma$: discount (`discount` = 0.95).
- $d$: latent dimension (`z_dim`), the fixed rank.
- $F_\theta^h(s,a,z)\in\mathbb R^d$: forward embedding, two heads $h\in\{1,2\}$.
- $B_\theta(s)\in\mathbb R^d$: backward embedding.
- $\pi_\theta(s,z)$: actor, a deterministic policy conditioned on the latent.
- $\Sigma_\theta$: $\mathbb E_\rho[B_\theta B_\theta^\top]$, the covariance of $B$. Estimated on $n_{\text{est}}$ states per step, and on $n_{\text{big}}$ states once per stage for $\Sigma_{\text{old}}$. A ridge $\kappa\,I$ is added before every inversion (`ridge`).
- $\theta_k,\ \theta_{\text{old}}$: model after stage $k$; $\theta_{\text{old}}=\theta_{k-1}$ is frozen during stage $k$.
- $A_\theta$: $\mathbb E_\rho[B_\theta B_{\text{old}}^\top]$, cross-covariance of the new and old backward embeddings.
- $T_\theta$: $A_\theta\Sigma_{\text{old}}^{-1}$, the matrix that carries an old latent into the new basis.
- $\bar z$: $\sqrt d\,z/\|z\|$, a latent rescaled to norm $\sqrt d$. This is the form that is fed to $F$ and $\pi$.
- $r_k,\ \bar r_k,\ m_{2,k}$: reward of task $k$; its normalized version $\bar r_k=r_k/\sqrt{m_{2,k}}$ with $m_{2,k}=\mathbb E_{D_k}[r_k^2]$.
- $D_k$: labeled set of task $k$: `n_labeled` states with their reward, fixed during the stage.
- $\nu$: distribution of latents used by the retention terms: $(1-\alpha)\,\text{uniform sphere}+\alpha\,\text{uniform mixture of the stored atoms }\bar z_j$, $j<k$.

## 1 Encoding a reward in an FB model

### Step 1.1 The factorization

For a policy $\pi$, the discounted occupancy of $\pi$ started from $(s,a)$ has a density with respect to $\rho$, and FB writes it as a rank-$d$ product:

$$
M^{\pi_z}(s,a,s') \;=\; F(s,a,z)^\top B(s') .
$$

$\pi_z$ is the policy $\pi(\cdot,z)$ of the actor.

### Step 1.2 Value of a reward

For a reward $r$, the $Q$-function of $\pi_z$ is the occupancy integrated against $r$:

$$
Q^{\pi_z}_r(s,a)=\mathbb E_{s'\sim\rho}\big[M^{\pi_z}(s,a,s')\,r(s')\big]=F(s,a,z)^\top\,\mathbb E_\rho\big[r(s')B(s')\big].
$$

If $z=z_r:=\mathbb E_\rho[rB]$, then $\pi_z$ is the policy built for $r$ and $Q_r^{\pi_{z}}(s,a)=F(s,a,z)^\top z$. A task is therefore described by $d$ numbers.

### Step 1.3 Reward reconstructed from a latent

Approximate $r$ by a linear function of $B$, $\hat r(s)=B(s)^\top w$. The least-squares problem and its solution are

$$
\min_w\ \mathbb E_\rho\big[(r-B^\top w)^2\big]\ \Rightarrow\ w^\star=\Sigma^{-1}\,\mathbb E_\rho[rB]=\Sigma^{-1}z_r .
$$

So the reward encoded by a latent $z$ is

$$
r_z(s)\;=\;B(s)^\top\Sigma^{-1}z .
$$

With an orthonormal $B$ ($\Sigma=I$) this is just $B^\top z$. The code keeps $\Sigma$ everywhere, so nothing relies on exact orthonormality.

Let $\Pi_\theta$ be the orthogonal projection in $L^2(\rho)$ onto $S_\theta=\mathrm{span}\{B_{\theta,1},\dots,B_{\theta,d}\}$. Then $\Pi_\theta r=B_\theta^\top\Sigma_\theta^{-1}z_\theta(r)$ with $z_\theta(r)=\mathbb E_\rho[rB_\theta]$, and the part of $r$ that FB cannot represent is

$$
\varepsilon(B_\theta;r):=\min_{w}\mathbb E_\rho\big[(r-B_\theta^\top w)^2\big]=\|r-\Pi_\theta r\|_\rho^2=\mathbb E_\rho[r^2]-z_\theta(r)^\top\Sigma_\theta^{-1}z_\theta(r). \tag{ε}
$$

FB acts as if the reward were $\Pi_\theta r$ (Step 6.1), and $\varepsilon$ depends on $\theta$ only through the subspace $S_\theta$. Everything below is about keeping $\varepsilon$ small, for the new task and for the old ones.

### Step 1.4 Normalization of the reward

Rewards of different tasks have different scales. On $D_k$ the reward is rescaled to unit second moment (no centering):

$$
\bar r_k=\frac{r_k}{\sqrt{m_{2,k}}},\qquad \mathbb E_{D_k}[\bar r_k^{\,2}]=1,\qquad z_k=\frac1{n}\sum_{i=1}^{n}\bar r_k(s_i)\,B_\theta(s_i),\quad s_i\in D_k .
$$

The scalar $m_{2,k}$ is kept for evaluation only.

## 2 The FB losses (unchanged from pretraining)

A batch is $(s_i,a_i,s'_i)_{i\le n}$ with a latent $z_i$ per row. $z_i$ is drawn as $\sqrt d\,\mathrm{normalize}(\mathcal N(0,I))$; with probability `mix_ratio` = 0.5 it is replaced by $\mathrm{sg}[B(s'_{\sigma(i)})]$ for a shuffled index $\sigma$, so that latents also come from the range of $B$. Bars denote the target networks (exponential moving average, `fb_target_tau`).

### Step 2.1 Bellman equation of the occupancy

The occupancy satisfies, in density form with respect to $\rho$,

$$
F(s,a,z)^\top B(s')\;=\;\frac{p(s'\mid s,a)}{\rho(s')}\;+\;\gamma\,\mathbb E_{s^+\sim p(\cdot\mid s,a)}\big[F(s^+,\pi_z(s^+),z)^\top B(s')\big].
$$

### Step 2.2 Temporal-difference loss

Write $M_{ij}^h=F^h(s_i,a_i,z_i)^\top B(s'_j)$ and $\bar M_{ij}=\min_h \bar F^h(s'_i,a'_i,z_i)^\top \bar B(s'_j)$, with $a'_i=\pi(s'_i,z_i)+$ clipped Gaussian noise (the actor is in the bootstrap: classical FB). Squaring the residual of Step 2.1 over $s'\sim\rho$ and expanding gives a quadratic term plus a linear term in $p/\rho$; the linear term is estimated with the observed next state, which is the diagonal. The batch estimate is

$$
\mathcal L_{FB}\;=\;\sum_{h=1}^{2}\Big[\ \frac{1}{2\,n(n-1)}\sum_{i\neq j}\big(M^h_{ij}-\gamma\,\bar M_{ij}\big)^2\;-\;\frac1n\sum_{i}M^h_{ii}\ \Big].
$$

Off-diagonal pairs $(i,j)$ play the role of $s'\sim\rho$ independent of $(s_i,a_i)$.

### Step 2.3 Orthonormality of B

For independent $s,s'\sim\rho$, $\mathbb E[(B(s)^\top B(s'))^2]=\operatorname{tr}\Sigma^2$ and $\mathbb E\|B\|^2=\operatorname{tr}\Sigma$. Hence

$$
\mathcal L_{\text{orth}}=\mathbb E\big[(B(s)^\top B(s'))^2\big]-2\,\mathbb E\|B(s)\|^2=\|\Sigma-I\|_F^2-d .
$$

Minimizing it pushes $\Sigma$ to $I$. The batch version uses $\mathrm{Cov}=BB^\top$: the mean of the squared off-diagonal entries minus twice the mean diagonal. Logged as `orth_loss`; `orth_l2` is $\|\hat\Sigma-I\|_F/\sqrt d$.

### Step 2.4 Auxiliary Q loss

With $\hat\Sigma=\mathrm{sg}[B^\top B/n]$ on the batch, the implicit reward of $z_i$ at the next state is $r_i=B(s'_i)^\top\hat\Sigma^{-1}z_i$ (Step 1.3):

$$
\mathcal L_Q=\sum_{h}\frac1n\sum_i\Big(F^h(s_i,a_i,z_i)^\top z_i-\big[r_i+\gamma\min_{h'}\bar F^{h'}(s'_i,a'_i,z_i)^\top z_i\big]\Big)^2 .
$$

### Step 2.5 Actor

$$
\mathcal L_\pi=-\,\mathbb E\Big[\min_h F^h\big(s,\pi(s,z),z\big)^\top z\Big]\quad\text{(DDPG).}
$$

With `bc_coef` $\beta>0$ (TD3+BC): $\ \mathcal L_\pi=-\dfrac{\mathbb E[Q]}{\mathrm{sg}\,\mathbb E|Q|}+\beta\,\mathbb E\|\pi(s,z)-a\|^2$.

### Step 2.6 Critic objective

$$
\mathcal L_{\text{critic}}=\mathcal L_{FB}+c_Q\,\mathcal L_Q+c_{\text{orth}}\,\mathcal L_{\text{orth}},\qquad c_Q=0.01,\ c_{\text{orth}}=1 .
$$

## 3 Stage k: the continual objective

$$
\boxed{\ \mathcal L_k=\mathcal L_{\text{critic}}+\lambda\,\mathcal L_{\text{rec}}(r_k)+\mu_B\,\mathcal R^B+\mu_Q\,\mathcal R^Q\ }
$$

The three added terms are computed from $B_\theta$, $F_\theta$, the frozen $\theta_{\text{old}}$ and the stored latents. The actor is trained on $\mathcal L_\pi$ as before.

### Step 3.1 New task: reconstruction loss $\mathcal L_{\text{rec}}$

On $D_k$ ($n$ states), compute the latent and the readout of the current $B$:

$$
\hat z=\frac1n\sum_i\bar r_i\,B_\theta(s_i),\qquad w=\mathrm{sg}\big[\Sigma_\theta^{-1}\hat z\big],\qquad \mathcal L_{\text{rec}}=\frac1n\sum_i\big(\bar r_i-B_\theta(s_i)^\top w\big)^2 .
$$

The stop-gradient covers the whole coefficient $w$. The gradient only moves $B_\theta(s_i)$ in the prediction. This is exact, not a heuristic:

> **Proposition (detached coefficient).** If $\theta\mapsto B_\theta$ is differentiable and $\Sigma_\theta\succ0$, then
> $\nabla_\theta\,\varepsilon(B_\theta;r)=\nabla_\theta\,\mathbb E_\rho\big[(r-B_\theta^\top\,\mathrm{sg}[\Sigma_\theta^{-1}z_\theta(r)])^2\big]$.

*Proof.* Let $\ell(\theta,w)=\mathbb E_\rho[(r-B_\theta^\top w)^2]$, strictly convex in $w$ with unique minimizer $w_\theta=\Sigma_\theta^{-1}z_\theta(r)$. Then $\varepsilon=\ell(\theta,w_\theta)$, so $\nabla_\theta\varepsilon=\partial_\theta\ell+\partial_w\ell\cdot\partial_\theta w_\theta$, and $\partial_w\ell(\theta,w_\theta)=0$. $\square$

The stop-gradient must cover the whole vector. Stopping only $z_\theta(r)$, or differentiating the closed form (ε) with $\Sigma_\theta$ frozen at $I$, does not give $\nabla\varepsilon$ when $\Sigma_\theta\neq I$.

$$
\nabla_\theta\mathcal L_{\text{rec}}=-\frac2n\sum_i\big(\bar r_i-B_\theta(s_i)^\top w\big)\,\nabla_\theta B_\theta(s_i)^\top w .
$$

Why this is the right quantity: if $\Sigma_\theta$ were the covariance of $B$ on $D_k$, then $\hat z^\top\Sigma^{-1}\hat z$ is the energy of $\bar r$ explained by $B$, and since $\frac1n\sum\bar r_i^2=1$,

$$
\mathcal L_{\text{rec}}=1-\hat z^\top\Sigma_\theta^{-1}\hat z=\varepsilon_k ,
$$

the fraction of the reward energy that $B$ cannot explain (logged as `new_reward_unexplained_fraction_training`).

**Cross-fitting** (`cross_fit=1`): split $D_k=D^{(1)}\cup D^{(2)}$; fit $\hat z$ and $w$ on one half and evaluate the loss on the other, then swap and average:

$$
\mathcal L_{\text{rec}}^{cf}=\tfrac12\Big[\ \mathrm{err}_{D^{(2)}}\big(w(D^{(1)})\big)+\mathrm{err}_{D^{(1)}}\big(w(D^{(2)})\big)\ \Big] .
$$

This removes the optimism of fitting and scoring on the same $n$ points.

### Step 3.2 The old family and the map $T$

Each old latent $z$ defines a reward $r_z^{\text{old}}(s)=B_{\text{old}}(s)^\top\Sigma_{\text{old}}^{-1}z$ (Step 1.3), with energy

$$
\mathbb E_\rho\big[(r_z^{\text{old}})^2\big]=z^\top\Sigma_{\text{old}}^{-1}\Sigma_{\text{old}}\Sigma_{\text{old}}^{-1}z=z^\top\Sigma_{\text{old}}^{-1}z .
$$

The new model can only represent rewards $B_\theta^\top w'$. The best such approximation of $r_z^{\text{old}}$ solves $\min_{w'}\mathbb E[(B_\theta^\top w'-B_{\text{old}}^\top w)^2]$ with $w=\Sigma_{\text{old}}^{-1}z$:

$$
w'=\Sigma_\theta^{-1}A_\theta\,w=\Sigma_\theta^{-1}A_\theta\Sigma_{\text{old}}^{-1}z .
$$

The latent of that approximation is $z'=\Sigma_\theta w'$, because the readout of a latent $z'$ is $\Sigma_\theta^{-1}z'$. Therefore

$$
z'=A_\theta\Sigma_{\text{old}}^{-1}z=T_\theta\,z,\qquad T_\theta=A_\theta\Sigma_{\text{old}}^{-1} .
$$

$T$ re-encodes the whole family $\{r_z\}$ at once. It is a general $d\times d$ matrix, not a rotation, and no relation between $B_\theta$ and $B_{\text{old}}$ is assumed. It is the only thing needed to carry stored latents forward. The point of this design: when $B_\theta$ changes to encode $r_k$, the latent coordinates change, and an old reward is allowed to get a different latent. What is retained is the reward and the behavior, each through the model's own encoding, not the latent itself.

### Step 3.3 Retention in latent space $\mathcal R^B$

The energy of $r_z^{\text{old}}$ that the new $B_\theta$ keeps is the energy of the projection $B_\theta^\top w'$:

$$
\mathbb E\big[(B_\theta^\top w')^2\big]=w'^\top\Sigma_\theta w'=z^\top T^\top\Sigma_\theta^{-1}T\,z .
$$

The energy that is lost is the difference:

$$
z^\top\Sigma_{\text{old}}^{-1}z-z^\top T^\top\Sigma_\theta^{-1}T z=z^\top M_\theta\,z,\qquad M_\theta=\Sigma_{\text{old}}^{-1}-T_\theta^\top\Sigma_\theta^{-1}T_\theta\ \succeq 0 .
$$

This is exactly $\varepsilon(B_\theta;r_z)=z^\top M_\theta z$ by (ε), since $z_\theta(r_z)=Tz$. And $M_\theta=\Sigma_{\text{old}}^{-1}\big(\Sigma_{\text{old}}-A_\theta^\top\Sigma_\theta^{-1}A_\theta\big)\Sigma_{\text{old}}^{-1}$, where the bracket is the Schur complement of the joint covariance of $(B_{\text{old}},B_\theta)$, hence $M_\theta\succeq0$; $\varepsilon(B_\theta;r_z)=0$ iff $r_z\in S_\theta$. The fraction lost is a ratio of quadratic forms, invariant to the scale of $z$. Averaging over $\nu$ gives the loss:

$$
\mathcal R^B=\mathbb E_{z\sim\nu}\left[\frac{z^\top M_\theta\,z}{z^\top\Sigma_{\text{old}}^{-1}z}\right]\in[0,1].
$$

It is zero when every old reward in the support of $\nu$ is exactly representable by the new $B_\theta$, and it is differentiated through $A_\theta$ and $\Sigma_\theta$ (batch estimates on $n_{\text{est}}$ states); $\Sigma_{\text{old}}$ is fixed for the stage. Logged as `old_rewards_lost_fraction`. Note that $\mathcal R^B$ only constrains the span of $B$. It does not constrain what $F$ and the policy do.

**Special case $\Sigma_{\text{old}}=\Sigma_\theta=I$.** Then $T=A$, $r_z=B_{\text{old}}^\top z$, $M=I-A^\top A$, $\varepsilon(B_\theta;r_z)=\|z\|^2-\|Az\|^2$, and with $\Omega=\mathbb E_\nu[zz^\top]$ and $\|z\|^2=d$,

$$
\mathcal R^B=1-\frac{\operatorname{tr}(\Omega\,A^\top A)}{d} .
$$

The singular values of $A$ are the cosines of the principal angles between $S_{\text{old}}$ and $S_\theta$: a singular value 1 means the corresponding old direction is fully preserved, a smaller one that it is partly lost.

### Step 3.4 Retention of the values $\mathcal R^Q$

A latent in the right basis is not enough: the forward embedding must also give the same value to the policy of the re-encoded reward. For $z\sim\nu$, set $z'=\mathrm{sg}[T_\theta z]$ and $\bar z'=\sqrt d\,z'/\|z'\|$. The old value of the old reward and the new value of its re-encoding are

$$
Q^{h}_{\text{old}}(s,a;z)=F^h_{\text{old}}(s,a,z)^\top z,\qquad Q^{h}_{\theta}(s,a;z')=F^h_\theta(s,a,\bar z')^\top z' .
$$

($F$ and the policy receive the direction $\bar z'$, the scale of $z'$ is carried by the dot product.) The loss matches them head by head:

$$
\mathcal R^Q=\frac12\sum_{h=1}^{2}\mathbb E_{(s,a)\sim D,\ z\sim\nu}\Big[\big(Q^h_\theta(s,a;z')-Q^h_{\text{old}}(s,a;z)\big)^2\Big] .
$$

Gradients flow into $F_\theta$ only; the old model and $z'$ are constants. Logged as `old_rewards_value_change`.

Two design choices. (i) Since $z_{\text{old}}(r_z)=z$ and $z_\theta(r_z)=z'$, the two numbers are the two models' values of the same reward $r_z$. The shrinkage of $\|z'\|$ when part of $r_z$ is no longer representable enters automatically through the dot product with the unnormalized $z'$. (ii) Values are matched, not the successor measures $M(s,a,\cdot)$: matching $M^{\pi_z}_{\text{old}}$ for all $z$ would force $S_{\text{old}}\subseteq S_\theta$ and leave no capacity for $r_k$ (Step 6.4). No separate policy term is needed: $\pi_\theta(\cdot,\bar z')$ is trained by $\mathcal L_\pi$ to maximize $F_\theta(s,a,\bar z')^\top z'$, so matching values gives matching greedy actions.

### Step 3.5 The distribution $\nu$

For each latent used by the retention terms, draw $u\sim U[0,1]$. If $u<\alpha$ and at least one atom exists, take $\sqrt d\,\mathrm{normalize}(z_j)$ for $j$ uniform in $\{1,\dots,k-1\}$. Otherwise take a uniform point on the sphere of radius $\sqrt d$. The sphere covers the whole family of rewards of the old $B$, the atoms concentrate the constraint on the tasks actually seen. Both $\mathcal R^B$ and $\mathcal R^Q$ (through $\bar z$) are invariant to the scale of a stored $z_j$, so $d$ numbers per task suffice.

### Step 3.5b Estimation and cost

$\Sigma_{\text{old}}$ is computed once per stage on $n_{\text{big}}$ states. At every step $\Sigma_\theta$ and $A_\theta$ are estimated on a batch of $n_{\text{est}}\gg d$ states: $\hat A_\theta=\frac1n\sum_iB_\theta(s_i)B_{\text{old}}(s_i)^\top$. The orthonormality loss keeps $\hat\Sigma_\theta$ well conditioned. Because $\Sigma_\theta^{-1}$ is nonlinear in the batch estimate, a large batch (or a running average) is needed. The cost is $O(n d^2)$ per step, independent of the number of previous tasks.

### Step 3.6 Gradient flow, in short

- $\mathcal L_{FB},\mathcal L_Q,\mathcal L_{\text{orth}}$: through $F_\theta,B_\theta$ (targets are constants).
- $\mathcal L_{\text{rec}}$: through $B_\theta(s_i)$ in the prediction only; the readout $w$ is a constant.
- $\mathcal R^B$: through $B_\theta$ via $A_\theta$ and $\Sigma_\theta$.
- $\mathcal R^Q$: through $F_\theta$ only.
- At the start of a stage: $\theta_{\text{old}}\leftarrow\theta$, target network $\leftarrow\theta$, fresh Adam states.

## 4 End of stage: what is stored

### Step 4.1 Carry the old latents forward

On a fresh batch of $n_{\text{big}}$ states, with $\theta_k$ the model that just finished and $\theta_{k-1}$ the frozen one:

$$
T=\Big(\tfrac1{n_{\text{big}}}\sum_s B_{\theta_k}(s)B_{\theta_{k-1}}(s)^\top\Big)\Sigma_{k-1}^{-1},\qquad z_j\leftarrow T\,z_j\quad(j<k).
$$

After several stages, the latent of task $j$ is a product of such maps: $z_j^{(K)}=T_K\cdots T_{j+1}\,z_j^{(j)}$.

### Step 4.2 Add the new task

$$
z_k=\frac1n\sum_{i}\bar r_k(s_i)\,B_{\theta_k}(s_i),\quad s_i\in D_k .
$$

### Step 4.3 Memory

One vector $z_j\in\mathbb R^d$ and one scalar $m_{2,j}$ per task. No data, no model copy, no $F$ is kept beyond the current $\theta_{\text{old}}$.

## 5 Evaluation metrics

### Step 5.1 Unexplained fraction $\varepsilon$

For a latent $z$, the reconstruction $\hat r=B^\top(\Sigma+\kappa I)^{-1}z$ is scored on held-out states ($\Sigma$ on separate states):

$$
\varepsilon=\frac{\mathbb E_{\text{hold}}\big[(\bar r-\hat r)^2\big]}{\mathbb E_{\text{hold}}[\bar r^{\,2}]},\qquad \text{rew\_corr}=\operatorname{corr}_{\text{hold}}(\bar r,\hat r) .
$$

$\varepsilon=0$ is perfect, $\varepsilon=1$ is no better than predicting zero.

### Step 5.2 Fresh and tracked latents

- **fresh**: $z$ is recomputed from 10k fresh labels with the current $B$, $z=\frac1n\sum\bar r\,B_\theta$. It measures what the current model can represent.
- **tracked**: $z$ is the stored latent $z_j$, carried through the $T$'s. It measures what the model remembers. The scalar $m_{2,j}$ puts the reward on the scale of $\bar r_j$ for scoring. It is never used in training.

### Step 5.3 Latent shrinkage

$$
\text{z\_shrink}=\mathbb E_{z\sim\nu}\left[\frac{\|T z\|}{\|z\|}\right]
$$

(logged as `old_latent_length_ratio`.) It is below 1 when part of the old rewards cannot be expressed in the new basis. It is not the same quantity as $\mathcal R^B$: it measures the length of $Tz$, not the energy retained.

### Step 5.4 Return

The actor is rolled out with the direction $\bar z$ of the latent, and the true environment reward is summed over the episode. The return is the quantity of interest; $\varepsilon$ is a diagnostic of the encoding.

## 6 Analysis: what the method guarantees

Let $\nu^\pi$ be the discounted state-occupancy measure of $\pi$ (total mass $1/(1-\gamma)$), $J_r(\pi)=\int r\,d\nu^\pi$, and $C(\pi)=\|d\nu^\pi/d\rho\|_{L^2(\rho)}$ the coverage constant, assumed finite.

### Step 6.1 Suboptimality: why $\varepsilon$ matters

> **Proposition.** Let $\hat r=\Pi_\theta r$, $\hat\pi=\pi_\theta(\cdot,\bar z_\theta(r))$, $\pi^\star$ optimal for $r$, and $\Delta_{\hat r}(\hat\pi)=\max_\pi J_{\hat r}(\pi)-J_{\hat r}(\hat\pi)$. Then

$$
J_r(\pi^\star)-J_r(\hat\pi)\;\le\;\Delta_{\hat r}(\hat\pi)+\big(C(\pi^\star)+C(\hat\pi)\big)\sqrt{\varepsilon(B_\theta;r)} .
$$

*Proof.* $J_r(\pi)=J_{\hat r}(\pi)+\langle\nu^\pi,r-\hat r\rangle$, so the gap equals $[J_{\hat r}(\pi^\star)-J_{\hat r}(\hat\pi)]+\langle\nu^{\pi^\star}-\nu^{\hat\pi},r-\hat r\rangle$. The bracket is at most $\Delta_{\hat r}(\hat\pi)$. The inner product is at most $(C(\pi^\star)+C(\hat\pi))\|r-\hat r\|_\rho$ by Cauchy–Schwarz. $\square$

The first term is the usual FB extraction error for $\hat r$. The second is controlled by $\varepsilon$ and is invisible to FB's own critic, because $z_\theta(r)=z_\theta(\hat r)$ and hence $F_\theta^\top z_\theta(r)=F_\theta^\top z_\theta(\hat r)$. This is why $\varepsilon$ is optimized through $B_\theta$.

### Step 6.2 Ceiling for peaky rewards

> **Proposition.** Assume $\|B_\theta(s)\|=\sqrt d$ for all $s$ and $\Sigma_\theta\succeq\lambda I$. Then for every $r\in L^2(\rho)$,

$$
1-\frac{\varepsilon(B_\theta;r)}{\mathbb E_\rho[r^2]}\;\le\;\frac d\lambda\,\frac{(\mathbb E_\rho|r|)^2}{\mathbb E_\rho[r^2]} .
$$

For an indicator reward on a set of $\rho$-mass $p$ and $\Sigma_\theta=I$, this reads $\varepsilon/\mathbb E_\rho[r^2]\ge1-dp$.

*Proof.* $\|z_\theta(r)\|\le\mathbb E_\rho[|r|\,\|B_\theta\|]=\sqrt d\,\mathbb E_\rho|r|$ and $z_\theta(r)^\top\Sigma_\theta^{-1}z_\theta(r)\le\|z_\theta(r)\|^2/\lambda$. Conclude with (ε). $\square$

Rewards supported on a fraction $p<1/d$ of the state mass are essentially unrepresentable at rank $d$, whatever the optimization. This is a ceiling, not a prediction.

### Step 6.3 Retention of past tasks across stages

Let $z_j^{(l)}$ be the vector stored for task $j$ after stage $l\ge j$, $M_l$ the matrix of Step 3.3 between $\theta_l$ and $\theta_{l-1}$, and

$$
\rho_{j,l}=\frac{z_j^{(l-1)\top}M_l\,z_j^{(l-1)}}{z_j^{(l-1)\top}\Sigma_{\theta_{l-1}}^{-1}z_j^{(l-1)}}
$$

the integrand of $\mathcal R^B$ at the atom of task $j$.

> **Proposition.** For every $j\le k$,

$$
\sqrt{\varepsilon(B_{\theta_k};\bar r_j)}\;\le\;\sqrt{\varepsilon(B_{\theta_j};\bar r_j)}+\sum_{l=j+1}^{k}\sqrt{\rho_{j,l}} .
$$

*Proof.* Let $\tilde r_j^{(l)}=B_{\theta_l}^\top\Sigma_{\theta_l}^{-1}z_j^{(l)}$. By construction $\tilde r_j^{(j)}=\Pi_{\theta_j}\bar r_j$ and $z_j^{(l)}=\mathbb E_\rho[B_{\theta_l}\tilde r_j^{(l-1)}]$, hence $\tilde r_j^{(l)}=\Pi_{\theta_l}\tilde r_j^{(l-1)}$. Then $\|\tilde r_j^{(l-1)}-\tilde r_j^{(l)}\|_\rho^2=\varepsilon(B_{\theta_l};\tilde r_j^{(l-1)})=\rho_{j,l}\,\|\tilde r_j^{(l-1)}\|_\rho^2\le\rho_{j,l}$, by Step 3.3 and because projections do not increase the norm of $\bar r_j$, which is 1. Also $\|\bar r_j-\tilde r_j^{(j)}\|_\rho^2=\varepsilon(B_{\theta_j};\bar r_j)$. Since $\tilde r_j^{(k)}\in S_{\theta_k}$, $\sqrt{\varepsilon(B_{\theta_k};\bar r_j)}\le\|\bar r_j-\tilde r_j^{(k)}\|_\rho$, and the triangle inequality telescopes. $\square$

Without labels, the part of $r_j$ outside $S_{\theta_j}$ is unrecoverable, and the drift accumulates by at most one term per stage. This is the bound that connects $\mathcal R^B$ to the error of the *tracked* reward.

### Step 6.4 Capacity at fixed rank

> **Proposition.** Let $r_1,\dots,r_K\in L^2(\rho)$, $G_{ij}=\langle r_i,r_j\rangle_\rho$ with eigenvalues $\lambda_1\ge\dots\ge\lambda_K$. For every $d$-dimensional subspace $S$,

$$
\sum_{j}\varepsilon_S(r_j)\;\ge\;\sum_{i>d}\lambda_i(G),
$$

with equality iff the span of the top-$d$ singular functions of $(r_1,\dots,r_K)$ is realizable by a pointwise-normalized $B_\theta$. In particular all $K$ rewards are exactly representable at rank $d$ only if $\operatorname{rank}(G)\le d$.

*Proof.* Let $R:\mathbb R^K\to L^2(\rho)$, $R\alpha=\sum_j\alpha_jr_j$, so $R^*R=G$. Then $\sum_j\|(I-\Pi_S)r_j\|_\rho^2=\|(I-\Pi_S)R\|_{HS}^2\ge\sum_{i>d}\sigma_i(R)^2$ by Eckart–Young–Mirsky, with $\sigma_i(R)^2=\lambda_i(G)$. $\square$

At fixed $d$, learning $r_k\notin\operatorname{span}\{r_1,\dots,r_{k-1}\}$ once the rank is saturated necessarily costs the representability of some old direction. The weights $\Omega$ (through $\alpha$ and the atoms $z_j$) decide which old directions are sacrificed first.

Open points for the next version: justify $\mathcal R^Q$ as the exact condition for value preservation (the analysis above covers the span of $B$, not $F$ and the policy), and add the fine-tuning modes (`finetune=actor|critic|reward_critic`), which use a fixed $z_{\text{cond}}=\sqrt d\,\mathrm{normalize}(z_k)$ and the readout $w_q=\Sigma_{\text{old}}^{-1}z_{\text{cond}}$.
