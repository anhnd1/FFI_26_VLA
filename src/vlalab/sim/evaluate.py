"""LIBERO closed-loop evaluation against a running policy server.

Per task: one env (seeded), then per episode: reset -> fixed initial state -> `num_steps_wait` dummy steps while
objects settle -> receding-horizon control: query a chunk of H actions, execute the first `replan_steps`, re-query.
Success = LIBERO's goal predicate (env returns done) within max_steps; otherwise failure.

Usage: python -m vlalab.sim.evaluate --config arm.json --port 8000 --out OUT_DIR
Outputs: see sim/recording.py.
"""
from __future__ import annotations

import argparse
import collections
import logging
import os
import time

import imageio
import numpy as np
from openpi_client import websocket_client_policy

from vlalab.config import ArmConfig
from vlalab.sim import libero_env
from vlalab.sim import recording

logger = logging.getLogger("vlalab.evaluate")


def run_episode(env, client, init_state, prompt: str, cfg, uid: int, recorder: recording.EpisodeRecorder):
    """One episode. Returns (success, env_steps, frames)."""
    ev = cfg.eval
    max_steps = ev.max_steps or libero_env.MAX_STEPS[ev.suite]
    env.reset()
    obs = env.set_init_state(init_state)
    for _ in range(ev.num_steps_wait):
        obs, _, done, _ = env.step(libero_env.DUMMY_ACTION)

    plan = collections.deque()
    frames, done, t = [], False, 0
    while t < max_steps:
        try:
            element = libero_env.policy_observation(obs, prompt, ev.resize)
            frames.append(element["observation/image"])
            if not plan:
                request = dict(element)
                request["meta/episode_uid"] = uid
                request["meta/call_in_episode"] = recorder.n_calls
                response = client.infer(request)
                chunk = response["actions"]
                recorder.add_call(response["call_index"], t, element, chunk)
                plan.extend(chunk[: ev.replan_steps])
            obs, _, done, _ = env.step(plan.popleft().tolist())
            t += 1
            if done:
                break
        except Exception:  # a crashed step ends the episode as a failure, the evaluation goes on
            logger.exception("episode aborted at step %d", t)
            break
    return bool(done), t, frames


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", force=True)

    cfg = ArmConfig.load(args.config)
    ev = cfg.eval
    out = args.out
    for sub in ("episodes", "videos"):
        os.makedirs(os.path.join(out, sub), exist_ok=True)
    cfg.save(os.path.join(out, "config.json"))
    table = recording.EpisodeTable(os.path.join(out, "episodes.csv"))

    np.random.seed(ev.seed)
    suite = libero_env.task_suite(ev.suite)
    task_ids = ev.tasks if ev.tasks is not None else list(range(suite.n_tasks))
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    start, n, successes = time.time(), 0, 0
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        env, prompt = libero_env.make_env(task, ev.env_resolution, ev.seed)
        for episode_idx in range(ev.start_trial, ev.start_trial + ev.trials):
            recorder = recording.EpisodeRecorder(ev.record_images)
            uid = recording.episode_uid(task_id, episode_idx)
            success, env_steps, frames = run_episode(env, client, init_states[episode_idx], prompt, cfg, uid, recorder)
            name = recording.episode_name(task_id, episode_idx)
            row = {
                "suite": ev.suite, "task_id": task_id, "task": prompt, "episode_idx": episode_idx,
                "success": int(success), "env_steps": env_steps, "n_calls": recorder.n_calls,
            }
            meta = dict(row, prompt=prompt, replan_steps=ev.replan_steps)
            recorder.save(os.path.join(out, "episodes", name + ".npz"), meta)
            if ev.save_video:
                video = os.path.join(out, "videos", "%s_%s.mp4" % (name, "success" if success else "failure"))
                imageio.mimwrite(video, [np.asarray(x) for x in frames], fps=10)
            table.add(row)
            n, successes = n + 1, successes + int(success)
            logger.info("task %d ep %d: %s in %d steps, %d calls | total %d/%d", task_id, episode_idx,
                        "success" if success else "failure", env_steps, recorder.n_calls, successes, n)
    table.close()
    recording.write_json(os.path.join(out, "summary.json"), {
        "arm": cfg.name, "suite": ev.suite, "episodes": n, "successes": successes,
        "success_rate": successes / max(n, 1), "rollout_s": round(time.time() - start, 1),
    })
    logger.info("success rate %d/%d = %.1f%%", successes, n, 100.0 * successes / max(n, 1))


if __name__ == "__main__":
    main()
