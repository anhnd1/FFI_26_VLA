"""Build every results table and figure from runs/ into results/.

Usage: python -m vlalab.analysis.build            (local; reads runs/<exp>/<arm>/, writes results/)

results/tables/   arms.md (+csv)        one row per arm: success with 95% CI, calls, rollout time, latency, VRAM
                  paired.md             each config vs the baseline on the same initial states (exact McNemar)
                  per_task.md           per-episode outcomes, task x config
                  episodes.csv          per-episode trajectory features (analysis/features.py)
                  by_outcome.md         features of successful vs failed episodes, all tasks and per failing task
results/figures/  success.png, latency.png, outcomes.png, features.png,
                  traj_<exp>_<arm>_<episode>.png and frames_<...>.png for every failed libero_10 episode
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vlalab.analysis import features as F
from vlalab.analysis import stats as S
from vlalab.analysis import style
from vlalab.analysis.runs import RESULTS, Arm, load_all, markdown, vram_peak_gb

TABLES, FIGURES = RESULTS / "tables", RESULTS / "figures"
BASELINE = "j1/base_t0"  # 10 steps, replan 5, initial states 0-1: the reference every ablation is paired with


def short_task(task: str, n: int = 38) -> str:
    return task if len(task) <= n else task[: n - 1] + "…"


def write_table(df: pd.DataFrame, name: str, intro: str = "") -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES / f"{name}.csv", index=False)
    (TABLES / f"{name}.md").write_text((intro + "\n\n" if intro else "") + markdown(df) + "\n")


# ---------------------------------------------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------------------------------------------
def arms_table(arms: list[Arm]) -> pd.DataFrame:
    rows = []
    for a in arms:
        ep = a.episodes
        k, n = int(ep.success.sum()), len(ep)
        lo, hi = S.wilson(k, n)
        lat = a.latency
        # arm i ran on GPU i; arms are listed alphabetically, which matches the order in experiments/*.yaml
        gpu = list(sorted(p.name for p in a.dir.parent.iterdir() if (p / "episodes.csv").exists())).index(a.name)
        rows.append({
            "run": a.key, "suite": a.config["eval"]["suite"], "config": a.label, "initial states": a.trials,
            "success": f"{k}/{n}", "rate %": round(100 * k / n, 1), "95% CI": f"{100 * lo:.0f}-{100 * hi:.0f}",
            "calls/episode": round(ep.n_calls.mean(), 1),
            "steps/success": round(ep[ep.success == 1].env_steps.mean(), 0),
            "rollout min": round(a.summary["rollout_s"] / 60, 1),
            "client p50 ms": round(lat.client_infer_ms.median()) if lat is not None else None,
            "server p50 ms": round(lat.server_infer_ms.median()) if lat is not None else None,
            "peak VRAM GiB": vram_peak_gb(a.exp).get(gpu),
        })
    return pd.DataFrame(rows)


def paired_table(arms: list[Arm]) -> pd.DataFrame:
    base = next(a for a in arms if a.key == BASELINE)
    key = ["task_id", "episode_idx"]
    rows = []
    for a in arms:
        if a.key == BASELINE or a.config["eval"]["suite"] != base.config["eval"]["suite"]:
            continue
        m = base.episodes.merge(a.episodes, on=key, suffixes=("_base", "_cfg"))
        if m.empty:
            continue
        only_b, only_c, p = S.mcnemar_exact(m.success_base.tolist(), m.success_cfg.tolist())
        rows.append({"config": f"{a.label} ({a.key})", "paired episodes": len(m),
                     "baseline only": only_b, "config only": only_c, "exact p": round(p, 3)})
    return pd.DataFrame(rows)


def per_task_table(arms: list[Arm]) -> pd.DataFrame:
    l10 = [a for a in arms if a.config["eval"]["suite"] == "libero_10"]
    tasks = l10[0].episodes.drop_duplicates("task_id").set_index("task_id").task
    out = pd.DataFrame({"task": [short_task(t, 60) for t in tasks]}, index=tasks.index)
    for a in l10:
        ep = a.episodes.sort_values("episode_idx")
        out[f"{a.label} [{a.trials}]"] = ep.groupby("task_id").success.apply(
            lambda s: " ".join("✓" if x else "✗" for x in s))
    return out.reset_index()


def episode_features(arms: list[Arm]) -> pd.DataFrame:
    rows = []
    for a in arms:
        for r in a.episodes.itertuples():
            f = F.episode_features(a.episode(r.task_id, r.episode_idx))
            rows.append({"run": a.key, "config": a.label, "suite": r.suite, "task": r.task, **f})
    return pd.DataFrame(rows)


def by_outcome_table(feats: pd.DataFrame) -> str:
    cols = ["n_calls", "grasp_attempts", "releases", "gripper_switches", "path_m", "stall_frac", "disagree_mean",
            "disagree_p90"]
    l10 = feats[feats.suite == "libero_10"]
    parts = []

    def summarize(df, title):
        g = df.groupby("success")[cols].mean().round(3)
        g.insert(0, "episodes", df.groupby("success").size())
        g.index = g.index.map({0: "failure", 1: "success"})
        parts.append(f"### {title}\n\n" + markdown(g.reset_index()))

    summarize(l10, "All libero_10 episodes, all configs")
    for task in l10[l10.success == 0].task.unique():
        summarize(l10[l10.task == task], f"Task: {task}")
    return "\n\n".join(parts) + "\n"


# ---------------------------------------------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------------------------------------------
def fig_success(table: pd.DataFrame) -> None:
    t = table[table.suite == "libero_10"].iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(t) + 1.2))
    y = np.arange(len(t))
    lo = np.array([float(c.split("-")[0]) for c in t["95% CI"]])
    hi = np.array([float(c.split("-")[1]) for c in t["95% CI"]])
    ax.hlines(y, lo, hi, color=style.SERIES[0], linewidth=2, alpha=0.45)
    ax.plot(t["rate %"], y, "o", color=style.SERIES[0], markersize=8)
    for yi, (r, s) in enumerate(zip(t["rate %"], t["success"])):
        ax.annotate(s, (r, yi), xytext=(8, 6), textcoords="offset points", fontsize=8, color=style.TEXT_2)
    ax.axvline(92.4, color=style.TEXT_2, linewidth=1, linestyle="--")
    ax.annotate("paper: 92.4%", (92.4, -0.45), xytext=(-4, 0), textcoords="offset points", ha="right",
                va="center", fontsize=8, color=style.TEXT_2)
    ax.set_ylim(-0.7, len(t) - 0.5)
    ax.set_yticks(y, [f"{c}  [{s}]" for c, s in zip(t.config, t["initial states"])])
    ax.set_xlim(50, 101)
    ax.set_xlabel("success rate % (dot) with 95% Wilson interval (line)")
    ax.set_title("LIBERO-Long success per configuration")
    style.save(fig, FIGURES / "success.png")


def fig_latency(arms: list[Arm]) -> None:
    l10 = [a for a in arms if a.latency is not None and a.config["eval"]["suite"] == "libero_10"]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(l10) + 1.2))
    for i, a in enumerate(l10[::-1]):
        v = a.latency.client_infer_ms / 1000
        ax.plot(v, np.full(len(v), i) + np.random.default_rng(i).uniform(-0.12, 0.12, len(v)), "o",
                color=style.SERIES[0], alpha=0.35, markersize=5)
        ax.plot([v.median()], [i], "|", color=style.TEXT, markersize=16, markeredgewidth=2)
        ax.annotate(f"p50 {v.median():.2f} s", (v.median(), i), xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=8, color=style.TEXT_2)
    ax.set_yticks(range(len(l10)), [f"{a.label}  ({a.key})" for a in l10[::-1]])
    ax.set_xlabel("client-side latency per policy call, s (10 benchmark calls; bar = median)")
    ax.set_title("Policy call latency on a Tesla T4")
    style.save(fig, FIGURES / "latency.png")


def fig_outcomes(arms: list[Arm]) -> None:
    l10 = [a for a in arms if a.config["eval"]["suite"] == "libero_10"]
    tasks = l10[0].episodes.drop_duplicates("task_id").set_index("task_id").task
    fig, ax = plt.subplots(figsize=(1.0 * len(l10) + 5, 0.42 * len(tasks) + 1.3))
    for j, a in enumerate(l10):
        for r in a.episodes.itertuples():
            k = r.episode_idx - a.config["eval"]["start_trial"]
            x, y = j + (k - 0.5) * 0.3, r.task_id
            ok = bool(r.success)
            ax.plot(x, y, "o" if ok else "X", color=style.SUCCESS if ok else style.FAILURE,
                    markersize=9 if ok else 11)
    ax.set_xticks(range(len(l10)), [f"{a.label}\n[{a.trials}]" for a in l10], fontsize=8)
    ax.set_yticks(list(tasks.index), [short_task(t, 48) for t in tasks], fontsize=8)
    ax.invert_yaxis()
    ax.grid(False)
    ax.plot([], [], "o", color=style.SUCCESS, label="success")
    ax.plot([], [], "X", color=style.FAILURE, label="failure")
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Every libero_10 episode (two initial states per cell)")
    style.save(fig, FIGURES / "outcomes.png")


def fig_features(feats: pd.DataFrame) -> None:
    l10 = feats[feats.suite == "libero_10"]
    cols = [("grasp_attempts", "sustained grasps"), ("gripper_switches", "gripper command switches"),
            ("stall_frac", "fraction of calls stalled"),
            ("disagree_mean", "mean re-plan disagreement"), ("n_calls", "policy calls")]
    fig, axes = plt.subplots(1, len(cols), figsize=(3.0 * len(cols), 3.2))
    for ax, (c, title) in zip(axes, cols):
        for x, (ok, color, marker) in enumerate([(1, style.SUCCESS, "o"), (0, style.FAILURE, "X")]):
            v = l10[l10.success == ok][c].dropna()
            ax.plot(x + np.random.default_rng(x).uniform(-0.15, 0.15, len(v)), v, marker, color=color,
                    alpha=0.5 if ok else 0.9, markersize=5 if ok else 8)
            if len(v):
                ax.plot([x - 0.25, x + 0.25], [v.median()] * 2, color=style.TEXT, linewidth=2)
        ax.set_xticks([0, 1], [f"success\n(n={int((l10.success == 1).sum())})",
                               f"failure\n(n={int((l10.success == 0).sum())})"])
        ax.set_title(title, fontsize=10)
    fig.suptitle("Trajectory features, all libero_10 episodes (black bar = median)", x=0.01, ha="left",
                 fontweight="bold")
    fig.tight_layout()
    style.save(fig, FIGURES / "features.png")


def fig_trajectory(a: Arm, task_id: int, idx: int, ref: tuple[Arm, int] | None = None) -> None:
    """Episode (optionally next to a reference episode of the same task): eef position, gripper command, disagreement."""
    eps = [(a, idx)] + ([ref] if ref is not None else [])
    fig, axes = plt.subplots(3, len(eps), figsize=(6 * len(eps), 7), sharex="col", squeeze=False,
                             gridspec_kw={"height_ratios": [2.2, 1, 1.2]})
    for col, (arm, i) in enumerate(eps):
        ep = arm.episode(task_id, i)
        outcome = "success" if ep["meta"]["success"] else "failure"
        t, r = ep["t"], ep["meta"]["replan_steps"]
        pos = ep["state"][:, :3]
        ax = axes[0, col]
        for d, name in enumerate("xyz"):
            ax.plot(t, pos[:, d] - pos[0, d], color=style.SERIES[d], label=name)
            ax.annotate(name, (t[-1], pos[-1, d] - pos[0, d]), xytext=(4, 0), textcoords="offset points",
                        color=style.TEXT_2, fontsize=9, va="center")
        stall = F.stalled(ep)
        for k in np.flatnonzero(stall):
            ax.axvspan(t[k] - r * 4, t[k], color=style.GRID, alpha=0.6, linewidth=0)
        ax.set_ylabel("eef position - start, m")
        src = "" if arm is a else f" ({arm.key}, {arm.label})"
        ax.set_title(f"{outcome}{src}: initial state {i}, {ep['meta']['env_steps']} steps", fontsize=10)
        if col == 0:
            ax.legend(loc="upper left", ncols=3)
        acts = F.executed_actions(ep)
        ax = axes[1, col]
        ax.step(np.arange(len(acts)), acts[:, 6], where="post", color=style.SERIES[6])
        ax.set_yticks([-1, 1], ["open", "close"])
        ax.set_ylabel("gripper cmd")
        grasps, _ = F.gripper_events(ep)
        ax.set_title(f"{len(grasps)} sustained grasps, {len(F.gripper_switches(ep))} command switches", fontsize=9,
                     color=style.TEXT_2, fontweight="normal")
        ax = axes[2, col]
        dis = F.replan_disagreement(ep)
        ax.plot(t[1:], dis, color=style.SERIES[0])
        ax.set_ylabel("re-plan\ndisagreement")
        ax.set_xlabel("policy env step (grey band: eef moved < 1 cm in the last 20 steps)")
    fig.suptitle(f"{a.key} ({a.label}): {ep['meta']['task']}", x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    style.save(fig, FIGURES / f"traj_{a.exp}_{a.name}_{task_id:02d}_ep{idx:02d}.png")


def fig_frames(a: Arm, task_id: int, idx: int, n: int = 6) -> None:
    """Agent-view policy inputs at n evenly spaced calls."""
    ep = a.episode(task_id, idx)
    if "image" not in ep:
        return
    ks = np.linspace(0, len(ep["t"]) - 1, n).round().astype(int)
    fig, axes = plt.subplots(1, n, figsize=(2.2 * n, 2.6))
    for ax, k in zip(axes, ks):
        ax.imshow(ep["image"][k])
        ax.set_title(f"step {ep['t'][k]}", fontsize=9, fontweight="normal")
        ax.axis("off")
    outcome = "success" if ep["meta"]["success"] else "failure"
    fig.suptitle(f"{a.key} {outcome}, initial state {idx}: {ep['meta']['task']}", x=0.01, ha="left", fontsize=10,
                 fontweight="bold")
    style.save(fig, FIGURES / f"frames_{a.exp}_{a.name}_{task_id:02d}_ep{idx:02d}.png")


def main() -> None:
    arms = load_all()
    print("arms:", [a.key for a in arms])
    table = arms_table(arms)
    n = 20
    write_table(table, "arms", f"Success with 95% Wilson intervals. With {n} episodes per arm and ~90% success, the "
                               f"smallest drop detectable at 80% power is about {100 * S.min_detectable_gap(n):.0f} "
                               f"points, so differences between configs below that are noise.")
    write_table(paired_table(arms), "paired", f"Each config vs {BASELINE} on the same initial states. "
                                              "Exact McNemar test on the discordant episodes.")
    write_table(per_task_table(arms), "per_task", "✓ success, ✗ failure, one mark per initial state.")
    feats = episode_features(arms)
    feats.to_csv(TABLES / "episodes.csv", index=False)
    (TABLES / "by_outcome.md").write_text(by_outcome_table(feats))
    print(table.to_string(index=False))

    fig_success(table)
    fig_latency(arms)
    fig_outcomes(arms)
    fig_features(feats)
    for a in arms:
        if a.config["eval"]["suite"] != "libero_10":
            continue
        ep = a.episodes
        for r in ep[ep.success == 0].itertuples():
            # reference success of the same task: same arm if possible, else the first arm that has one
            ref = None
            for b in [a] + [b for b in arms if b is not a and b.config["eval"]["suite"] == "libero_10"]:
                ok = b.episodes[(b.episodes.task_id == r.task_id) & (b.episodes.success == 1)].episode_idx
                if len(ok):
                    ref = (b, int(ok.iloc[0]))
                    break
            fig_trajectory(a, r.task_id, r.episode_idx, ref)
            fig_frames(a, r.task_id, r.episode_idx)
    # Successes with more sustained grasps than usual for their task: candidate recoveries from a failed grasp
    # (strength), plus the successes with the most gripper flicker (hesitation that resolved).
    feats = feats[feats.suite == "libero_10"].copy()
    typical = feats[feats.success == 1].groupby("task").grasp_attempts.median()
    feats["extra_grasps"] = feats.grasp_attempts - feats.task.map(typical)
    ok = feats[feats.success == 1]
    rec = ok[ok.extra_grasps >= 1].sort_values(["extra_grasps", "env_steps"], ascending=False)
    flick = ok.sort_values("gripper_switches", ascending=False).head(5)
    cols = ["run", "task", "episode_idx", "grasp_attempts", "extra_grasps", "gripper_switches", "env_steps"]
    (TABLES / "recoveries.md").write_text(
        f"Successful libero_10 episodes with more sustained grasps (closed >= {F.MIN_HOLD} steps) than the task's "
        f"median success: {len(rec)} of {len(ok)}. Look at the frames before calling one a recovery.\n\n"
        + markdown(rec[cols]) + "\n\nSuccesses with the most gripper command switches (flicker):\n\n"
        + markdown(flick[cols]) + "\n")
    by_key = {a.key: a for a in arms}
    for r in pd.concat([rec.head(3), flick.head(1)]).drop_duplicates(["run", "task_id", "episode_idx"]).itertuples():
        fig_trajectory(by_key[r.run], r.task_id, r.episode_idx)
        fig_frames(by_key[r.run], r.task_id, r.episode_idx, n=8)
    print("written:", sorted(p.name for p in FIGURES.iterdir()))


if __name__ == "__main__":
    main()
