"""vlalab: our own inference, evaluation and analysis code for pi0.5 on LIBERO.

Layout (each subpackage runs in a different Python environment, so they never import each other's heavy deps):
  config       shared experiment config (Python 3.8+, stdlib only)
  policy/      policy server side: checkpoint loading, I/O pipeline, flow sampler, websocket server, replay
               (Python 3.11, JAX; uses openpi only as a library for the network, weights and tokenizer)
  sim/         LIBERO side: rollouts, recording, latency benchmark (Python 3.8, robosuite/mujoco)
  analysis/    offline analysis of fetched runs: tables, statistics, trajectories, figures (local, CPU)
"""
