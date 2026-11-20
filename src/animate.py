"""
3D animation of a recorded run (default: data/nibiru_myth_run.npz written by nibiru_sweep.py).

Two 3D panels: inner solar system (heliocentric, +-4 AU) and the wide view (+-65 AU) showing
Nibiru's whole passage. Trails fade over the last `trail_years`; the camera slowly rotates.
Playback slows down while Nibiru is close to the Sun (watch the time label).

Usage:
  python animate.py                                  # -> outputs/animations/nibiru.mp4 (.gif without ffmpeg)
  python animate.py --run data/nibiru_myth_run.npz --frames 600 --fps 30 --out my_anim.mp4
  python animate.py --tmax 150                       # only animate the first 150 yr
"""
import argparse

import numpy as np
from paths import DATA, ANIMATIONS
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

PLANET_COLORS = plt.cm.tab10(np.arange(10))


def load_run(path, tmax=None):
    d = np.load(path, allow_pickle=False)
    t, r, names = d["t"], d["r"], [str(n) for n in d["names"]]
    if tmax is not None:
        keep = t <= tmax
        t, r = t[keep], r[keep]
    rel = r - r[:, [0]]          # heliocentric: Sun fixed at the origin
    return t, rel, names


NIBIRU_COLOR = "#00e5ff"


def frame_indices(t, rel, has_nibiru, frames, r_slow=1.5, r_fast=30.0):
    """
    Pick which recorded samples become frames. Without Nibiru: uniform in time.
    With Nibiru: playback slows near perihelion (frame density ~ r^-1.5, Kepler-like),
    clamped between r_slow and r_fast so the far-out part still advances.
    """
    n = min(frames, len(t))
    if not has_nibiru:
        return np.linspace(0, len(t) - 1, n).astype(int)
    r = np.clip(np.linalg.norm(rel[:, -1], axis=1), r_slow, r_fast)
    w = r**-1.5
    s = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(t))])
    return np.unique(np.searchsorted(s, np.linspace(0, s[-1], n)).clip(0, len(t) - 1))


def make_animation(t, rel, names, frames=600, fps=30, trail_years=(3.0, 40.0), out=str(ANIMATIONS / "nibiru.mp4")):
    N = rel.shape[1]
    has_nibiru = names[-1] == "Nibiru"
    idx = frame_indices(t, rel, has_nibiru, frames)                     # recorded sample shown in each frame
    dt_rec = t[1] - t[0]

    fig = plt.figure(figsize=(14, 7), facecolor="black")
    fig.subplots_adjust(left=0.0, right=0.94, top=0.95, bottom=0.06, wspace=0.08)
    panels = [
        dict(ax=fig.add_subplot(1, 2, 1, projection="3d"), lim=4.0, zlim=1.5,
             bodies=[i for i in range(1, N) if np.median(np.linalg.norm(rel[:, i], axis=1)) < 3 or names[i] == "Nibiru"],
             trail=trail_years[0], title="Inner solar system (±4 AU)"),
        dict(ax=fig.add_subplot(1, 2, 2, projection="3d"), lim=65.0, zlim=25.0,
             bodies=list(range(1, N)), trail=trail_years[1], title="Whole passage (±65 AU)"),
    ]

    def color(i):
        return NIBIRU_COLOR if names[i] == "Nibiru" else PLANET_COLORS[(i - 1) % 10]

    for P in panels:
        ax = P["ax"]
        ax.set_facecolor("black")
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color((0, 0, 0, 0))
            axis.label.set_color("0.7")
            axis.set_tick_params(colors="0.5", labelsize=7)
            axis._axinfo["grid"]["color"] = (0.3, 0.3, 0.3, 0.4)
        ax.set_xlim(-P["lim"], P["lim"])
        ax.set_ylim(-P["lim"], P["lim"])
        ax.set_zlim(-P["zlim"], P["zlim"])
        ax.set_box_aspect((1, 1, P["zlim"] / P["lim"]))
        ax.set_xlabel("x (AU)")
        ax.set_ylabel("y (AU)")
        ax.set_zlabel("z (AU)")
        ax.set_title(P["title"], color="white")
        ax.scatter([0], [0], [0], color="gold", s=80, depthshade=False)
        P["n_trail"] = max(2, int(P["trail"] / dt_rec))
        P["trails"], P["dots"] = {}, {}
        for i in P["bodies"]:
            lc = Line3DCollection([np.zeros((2, 3))], linewidths=1.5 if names[i] == "Nibiru" else 1.0)
            ax.add_collection3d(lc, autolim=False)
            P["trails"][i] = lc
            P["dots"][i], = ax.plot([], [], [], "o", color=color(i),
                                    markersize=7 if names[i] == "Nibiru" else 4, label=names[i])
        ax.legend(loc="upper left", fontsize=7, labelcolor="white", facecolor="black", edgecolor="0.3")

    time_text = fig.text(0.5, 0.04, "", color="white", ha="center", fontsize=12)
    nib_text = fig.text(0.5, 0.01, "", color=NIBIRU_COLOR, ha="center", fontsize=10)

    def update(f):
        k = idx[f]
        artists = []
        for P in panels:
            k0 = max(0, k - P["n_trail"])
            for i in P["bodies"]:
                pts = rel[k0:k + 1, i]
                inbox = np.all(np.abs(pts[:, :2]) <= P["lim"], axis=1) & (np.abs(pts[:, 2]) <= P["zlim"])
                pts = pts[inbox]                                          # 3D axes don't clip; do it by hand
                if len(pts) >= 2:
                    segs = np.stack([pts[:-1], pts[1:]], axis=1)
                    alpha = np.linspace(0.05, 0.9, len(segs))
                    rgba = np.tile(matplotlib.colors.to_rgba(color(i)), (len(segs), 1))
                    rgba[:, 3] = alpha
                    P["trails"][i].set_segments(segs)
                    P["trails"][i].set_color(rgba)
                    P["trails"][i].set_visible(True)
                else:
                    P["trails"][i].set_visible(False)   # set_segments([]) leaves the old 3D projection drawn
                x, y, z = rel[k, i]
                inside = abs(x) <= P["lim"] and abs(y) <= P["lim"] and abs(z) <= P["zlim"]
                P["dots"][i].set_data_3d([x] if inside else [], [y] if inside else [], [z] if inside else [])
                artists += [P["trails"][i], P["dots"][i]]
            P["ax"].view_init(elev=25, azim=-60 + 60 * f / len(idx))
        time_text.set_text(f"t = {t[k]:7.2f} yr")
        if has_nibiru:
            nib_text.set_text(f"Nibiru–Sun distance: {np.linalg.norm(rel[k, -1]):6.2f} AU")
        return artists

    anim = animation.FuncAnimation(fig, update, frames=len(idx), interval=1000 / fps, blit=False)
    if out.endswith(".mp4") and animation.writers.is_available("ffmpeg"):
        anim.save(out, writer=animation.FFMpegWriter(fps=fps, bitrate=4000), dpi=110,
                  savefig_kwargs={"facecolor": "black"})
    else:
        out = out.rsplit(".", 1)[0] + ".gif"
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=80,
                  savefig_kwargs={"facecolor": "black"})
    plt.close(fig)
    print(f"saved {out}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default=str(DATA / "nibiru_myth_run.npz"))
    ap.add_argument("--out", default=str(ANIMATIONS / "nibiru.mp4"))
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--tmax", type=float, default=None, help="only animate up to this time (yr)")
    args = ap.parse_args()
    t, rel, names = load_run(args.run, args.tmax)
    make_animation(t, rel, names, frames=args.frames, fps=args.fps, out=args.out)
