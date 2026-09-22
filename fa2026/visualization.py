from __future__ import annotations

import time

import matplotlib.pyplot as plt
from matplotlib import colors, patches
import numpy as np

from .scenarios import Goal


# Fixed visualization-only colors. Pheromone channels have no built-in meaning.
PHEROMONE_COLORS = (
    "#D55E00",  # vermillion
    "#0072B2",  # blue
    "#009E73",  # bluish green
    "#CC79A7",  # reddish purple
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#F0E442",  # yellow
    "#999999",  # gray
)


class Visualizer:
    """Small Matplotlib visualizer used by ``fa.run(..., visualize=True)``."""

    def __init__(self, engine):
        self.engine = engine
        self.fig, self.ax = plt.subplots(figsize=(10, 7))
        self._display_handle = None
        self._notebook = self._running_in_notebook()
        self._setup()

    @staticmethod
    def _running_in_notebook() -> bool:
        try:
            from IPython import get_ipython

            shell = get_ipython()
            return shell is not None and shell.__class__.__name__ == "ZMQInteractiveShell"
        except Exception:
            return False

    def _setup(self) -> None:
        e = self.engine
        h, w = e.scenario.grid_shape
        self.ax.set_aspect("equal")
        self.ax.set_xlim(0, w)
        self.ax.set_ylim(0, h)
        self.ax.set_xticks(np.arange(0, w + 1, 1), minor=True)
        self.ax.set_yticks(np.arange(0, h + 1, 1), minor=True)
        self.ax.grid(which="minor", linewidth=0.2, alpha=0.15)

        target = np.ma.masked_where(~e.target_mask, e.target_mask)
        self.ax.imshow(
            target,
            origin="lower",
            extent=(0, w, 0, h),
            interpolation="nearest",
            alpha=0.25,
            cmap="Greens",
            vmin=0,
            vmax=1,
            zorder=0,
        )

        self.phero_img = None
        if e.scenario.n_pheromones:
            self.phero_img = self.ax.imshow(
                self._pheromone_rgba(),
                origin="lower",
                extent=(0, w, 0, h),
                interpolation="nearest",
                zorder=1,
            )

        ys, xs = np.nonzero(e.walls)
        for y, x in zip(ys, xs):
            self.ax.add_patch(
                patches.Rectangle(
                    (x, y), 1, 1, facecolor="black", edgecolor="none", zorder=3
                )
            )

        positions = e.agent_positions
        self.agent_scatter = self.ax.scatter(
            positions[:, 0], positions[:, 1], s=10, c="crimson", zorder=5
        )

        self.cargo_patches: list[patches.Rectangle] = []
        for center, size in zip(e.cargo_positions, e.cargo_sizes):
            patch = patches.Rectangle(
                (center[0] - size / 2, center[1] - size / 2),
                size,
                size,
                facecolor="royalblue",
                edgecolor="navy",
                linewidth=1.5,
                zorder=4,
            )
            self.ax.add_patch(patch)
            self.cargo_patches.append(patch)

        self._update_title()
        self.fig.tight_layout()

    def _pheromone_rgba(self) -> np.ndarray:
        """Blend all pheromone channels for plotting only.

        Each channel is normalized independently by its current maximum, so a
        high-concentration channel does not automatically hide all others.
        """

        e = self.engine
        h, w = e.scenario.grid_shape
        weighted_rgb = np.zeros((h, w, 3), dtype=float)
        weight = np.zeros((h, w), dtype=float)

        for channel in range(e.scenario.n_pheromones):
            rgb = np.asarray(colors.to_rgb(PHEROMONE_COLORS[channel]), dtype=float)
            field = e.pheromone_fields[channel]
            maximum = float(np.max(field))
            if maximum <= 0.0:
                continue
            intensity = np.clip(field / maximum, 0.0, 1.0)
            weighted_rgb += intensity[..., None] * rgb
            weight += intensity

        rgba = np.zeros((h, w, 4), dtype=float)
        mask = weight > 0.0
        if np.any(mask):
            rgba[mask, :3] = weighted_rgb[mask] / weight[mask, None]
            rgba[..., 3] = 0.50 * np.clip(weight, 0.0, 1.0)
        return rgba

    def _update_title(self) -> None:
        e = self.engine
        mean_energy = float(np.mean(e.agent_energies)) if e.agents else 0.0
        parts = [f"turn {e.turn}", f"mean energy {mean_energy:.2f}"]

        if e.scenario.goal in (Goal.CARGOES, Goal.BOTH):
            delivered = sum(
                e._cargo_inside_target(i) for i in range(e.scenario.n_cargoes)
            )
            parts.append(f"cargo in target {delivered}/{e.scenario.n_cargoes}")

        if e.scenario.goal in (Goal.AGENTS, Goal.BOTH):
            gathered = sum(e._agent_inside_target(a.position) for a in e.agents)
            parts.append(f"agents in target {gathered}/{e.scenario.n_agents}")

        self.ax.set_title("  |  ".join(parts))

    def draw(self, *, delay: float = 0.0) -> None:
        e = self.engine
        self.agent_scatter.set_offsets(e.agent_positions)
        for patch, center, size in zip(
            self.cargo_patches, e.cargo_positions, e.cargo_sizes
        ):
            patch.set_xy((center[0] - size / 2, center[1] - size / 2))
            patch.set_width(size)
            patch.set_height(size)
        if self.phero_img is not None:
            self.phero_img.set_data(self._pheromone_rgba())
        self._update_title()

        if self._notebook:
            try:
                from IPython.display import display

                if self._display_handle is None:
                    self._display_handle = display(self.fig, display_id=True)
                else:
                    self._display_handle.update(self.fig)
            except Exception:
                self.fig.canvas.draw_idle()
        else:
            plt.show(block=False)
            self.fig.canvas.draw_idle()
            plt.pause(max(delay, 1e-6))

        if delay > 0 and self._notebook:
            time.sleep(delay)

    def finish(self) -> None:
        # Closing notebook figures prevents the inline backend from printing the
        # final frame a second time after the animation has already shown it.
        if self._notebook:
            plt.close(self.fig)
        else:
            self.fig.canvas.draw_idle()
