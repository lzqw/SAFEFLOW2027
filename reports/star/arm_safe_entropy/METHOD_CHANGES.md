# Arm safe entropy: implementation and development changes

Starting branch commit: `a251b7a` (P4.7 code plus all20 native results and
diagnosis). New branch: `codex/arm-safe-entropy-v1`. The user's dirty source
worktree and all historical navigation results are preserved. No rollback,
navigation resume or historical provenance repair is a prerequisite.

New independent arm runtime follows the existing bounded midpoint flow,
target-flow reward backups (4 stochastic target samples), two reward critic
updates, actor interval2, target actor/reward Polyak .001 and uniform+recent
reward/actor replay. Gaussian SAC uses conventional squashed-normal sampling,
log-probability correction, automatic temperature and twin Q backups. Its
implementation is visible in models.py, not labelled as an unmodified SB3 run.
No second behavior actor or action candidate reranking is introduced.

Before the first ability run, two explicit design corrections:

1. Remove independent uniform base actions from Brownian arm flow. Their
   variance 1/3 per axis exceeds the .04 reference variance and could mask the
   very endpoint directional covariance being tested. Initial base is zero;
   the SDE still samples stochastic actions and all behaviors/targets/evaluations
   use that one policy. Raw proposals are clipped to action bounds and clipping
   is logged; there is no separate safety action projection.
2. Match reference energy units: E_D=E_kin/.04,
   alpha_D=.04*alpha_kin and target_D=target_kin/.04. Initial alpha_kin=.02,
   target_kin=.25 per action coordinate; alpha_D=.0008 and total energy
   target=18.75. The legacy target12.5 per coordinate exceeds the flow's bounded
   kinetic maximum2 per coordinate at max velocity2, so it is not carried over.
   The original log-space automatic energy-dual update is retained.

Initial ability implementation has geometry_source=none, no global cost
optimization and no obstacle. Safety cost critic/margin heads and the full
Oracle/Learned matrix are **not yet implemented or claimed complete**.
Directional loss, inward drift and safety projection stay off in arm-v1.

Minimum validation only: three task clips and reset/cost/true-terminal checks,
unit-normal/trace/stop-gradient checks when geometry is implemented, and one
short learner save/resume. No historical matrix test suite or disaster-recovery
framework is added. Latest replay/checkpoints stay on A100, compact evidence
goes remote→host→GitHub.
