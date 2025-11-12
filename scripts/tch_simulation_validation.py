#!/usr/bin/env python3
# tch_simulation_validation.py
"""
Thermodynamic Cognitive Homeostasis (TCH) — Multi-Agent Simulator
=================================================================
Core simulation model implementing the TCH equations (3)–(5) from the paper.
Provides TCHAgent and TCHSimulator classes for reproducible numerical validation,
parameter sweeps, and publication-quality figure generation.

This implementation:
- Properly implements Eq. (5): Homeostasis = -∇_w[(S_cog - S_eq)²]
- Updates WEIGHTS via gradient, then recomputes S_cog as S_cog(w)
- Maintains full compatibility with paper's figures and results
- Uses fixed RNG seeds for full reproducibility

Generates:
  - phi_time_series_baseline.pdf : baseline Phi(t)
  - phi_alpha_comparison.pdf : Phi(t) for several alpha_homeo values
  - phi_phase_heatmap.pdf : final Phi as function of (coupling_S, alpha_homeo)
  - tch_summary_results.csv : small table with selected runs
  - phi_heatmap_matrix.csv : matrix used for heatmap

Design notes:
  - Uses fixed RNG seeds for reproducibility.
  - Uses matplotlib (one chart per figure). No seaborn.
  - Heatmap is plotted with pcolormesh to keep vector output in PDF.
"""

from __future__ import annotations
import os
import numpy as np
import matplotlib
# Ensure non-interactive backend for headless environments
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# -----------------------------
# Simulation model
# -----------------------------
class TCHAgent:
    def __init__(self, S_eq: float = 1.0, E0: float = 1.0, w_dim: int = 16, rng: np.random.RandomState = None):
        if rng is None:
            rng = np.random.RandomState(0)
        self.S_eq = float(S_eq)
        self.E = float(E0)
        self.w_dim = w_dim
        self.w = (rng.randn(w_dim) * 0.1).astype(float)
        # S_cog is a FUNCTIONAL of w: S_cog = 0.5 * ||w||² + S_eq
        self.S_cog = self.compute_S_cog()

    def compute_S_cog(self) -> float:
        """Compute cognitive entropy as functional of weights: S_cog = 0.5 * ||w||² + S_eq"""
        return 0.5 * float(np.sum(self.w**2)) + self.S_eq

    def compute_phi(self) -> float:
        """Order parameter φ = S_cog / S_eq"""
        return float(self.S_cog / (self.S_eq + 1e-12))

    def compute_dS_dw(self) -> np.ndarray:
        """Gradient of S_cog with respect to weights: ∂S/∂w = w"""
        return self.w.copy()


class TCHSimulator:
    """
    Lightweight TCH simulator for publication figures.
    All functional forms are intentionally simple and documented.
    """
    def __init__(self, N: int = 30, dt: float = 0.05, steps: int = 800,
                 coupling_S: float = 0.15, coupling_E: float = 0.03, seed: int = 0):
        self.N = int(N)
        self.dt = float(dt)
        self.steps = int(steps)
        self.kappa_S = float(coupling_S)
        self.kappa_E = float(coupling_E)
        self.rng = np.random.RandomState(int(seed))
        # create agents with independent RNG states but reproducible
        self.agents = [
            TCHAgent(S_eq=1.0, E0=1.0, w_dim=16,
                     rng=np.random.RandomState(self.rng.randint(0, 2**31 - 1)))
            for _ in range(self.N)
        ]
        self.history = {'Phi': [], 'mean_S': [], 'mean_E': []}

    @staticmethod
    def f_fun(E: float, phi: float) -> float:
        """Entropy production: increases with energy, suppressed by order"""
        return 0.12 * E * (1.0 - phi)

    @staticmethod
    def g_fun(L: float) -> float:
        """Dissipation proportional to current mismatch from equilibrium"""
        return 0.04 * L

    @staticmethod
    def h_fun(P: float) -> float:
        """Energy loss proportional to cognitive power P"""
        return 0.18 * P

    def couple(self):
        """Compute inter-agent diffusive coupling"""
        S_vals = np.array([a.S_cog for a in self.agents])
        E_vals = np.array([a.E for a in self.agents])
        S_mean = float(np.mean(S_vals))
        E_mean = float(np.mean(E_vals))
        dS_coup = self.kappa_S * (S_mean - S_vals)
        dE_coup = self.kappa_E * (E_mean - E_vals)
        return dS_coup, dE_coup

    def step_agent(self, agent: TCHAgent, dt: float, alpha_homeo: float, eta: float):
        """
        Single-agent dynamics for one Euler step:
          1. Compute gradient-based homeostasis on WEIGHTS (Eq. 5)
          2. Update weights via homeostatic gradient descent
          3. Recompute S_cog as functional of updated weights
          4. Compute cognitive power from weight velocity
          5. Update energy via Eq. (4)
        """
        # Current state
        phi = agent.compute_phi()
        L = abs(agent.S_cog - agent.S_eq)
        
        # Entropy dynamics (Eq. 3) WITHOUT direct homeostatic term
        dS_dt = self.f_fun(agent.E, phi) - self.g_fun(L)
        
        # WEIGHT UPDATE via homeostatic gradient (Eq. 5)
        # homeo_grad = -∇_w[(S_cog - S_eq)²] = -2*(S_cog - S_eq) * ∂S_cog/∂w
        dS_dw = agent.compute_dS_dw()
        homeo_grad = -2.0 * (agent.S_cog - agent.S_eq) * dS_dw
        
        # Weight velocity (cognitive work done for homeostasis)
        w_dot = alpha_homeo * eta * homeo_grad
        
        # Apply weight update
        agent.w += w_dot * dt
        
        # Recompute S_cog as functional of new weights (this is the KEY fix)
        agent.S_cog = agent.compute_S_cog()
        
        # Cognitive power: P = ||w_dot||² (Eq. 4)
        P = float(np.sum(w_dot**2))
        
        # Energy dynamics (Eq. 4)
        dE_dt = - self.h_fun(P)
        
        # Euler integration for energy
        agent.E += dE_dt * dt

        # Small stochastic drift in weights to break symmetry (reproducible)
        agent.w += 0.001 * self.rng.randn(*agent.w.shape)
        # Recompute S_cog after noise too
        agent.S_cog = agent.compute_S_cog()

        return {'dS': dS_dt * dt, 'dE': dE_dt * dt, 'phi': phi, 'P': P}

    def run(self, alpha_homeo: float = 1.0, eta: float = 0.02, verbose: bool = False):
        self.history = {'Phi': [], 'mean_S': [], 'mean_E': []}
        for t in range(self.steps):
            dS_coup, dE_coup = self.couple()
            for i, agent in enumerate(self.agents):
                # Step agent dynamics
                self.step_agent(agent, self.dt, alpha_homeo=alpha_homeo, eta=eta)
                # Apply coupling fluxes
                agent.S_cog += dS_coup[i] * self.dt
                agent.E += dE_coup[i] * self.dt

            # Record global metrics
            phis = np.array([a.compute_phi() for a in self.agents])
            self.history['Phi'].append(float(np.mean(phis)))
            self.history['mean_S'].append(float(np.mean([a.S_cog for a in self.agents])))
            self.history['mean_E'].append(float(np.mean([a.E for a in self.agents])))

            if verbose and (t % max(1, (self.steps // 8)) == 0):
                print(f"[run] t={t}/{self.steps} Phi={self.history['Phi'][-1]:.4f}")


# -----------------------------
# Figure generation (PDF)
# -----------------------------
def ensure_outdir(path: str = "figures"):
    os.makedirs(path, exist_ok=True)
    return path


def save_baseline_time_series(outdir: str, rng_seed: int = 1234):
    sim = TCHSimulator(N=30, dt=0.05, steps=800, coupling_S=0.15, coupling_E=0.03, seed=rng_seed)
    sim.run(alpha_homeo=1.2, eta=0.02)
    t = np.arange(sim.steps) * sim.dt
    fname = os.path.join(outdir, "phi_time_series_baseline.pdf")
    plt.figure(figsize=(6.5, 3.5))
    plt.plot(t, sim.history['Phi'])
    plt.xlabel("time")
    plt.ylabel(r"collective order parameter $\Phi$")
    plt.title(r"Baseline TCH: $\Phi(t)$")
    plt.tight_layout()
    plt.savefig(fname, format="pdf")
    plt.close()
    return fname, sim


def save_alpha_comparison(outdir: str, alphas=(0.5, 1.0, 2.0), rng_seed: int = 42):
    plt.figure(figsize=(6.5, 3.5))
    for a in alphas:
        sim = TCHSimulator(N=30, dt=0.05, steps=600, coupling_S=0.15, coupling_E=0.03, seed=rng_seed)
        sim.run(alpha_homeo=float(a), eta=0.02)
        t = np.arange(sim.steps) * sim.dt
        plt.plot(t, sim.history['Phi'], label=fr"$\alpha_{{\mathrm{{homeo}}}}={a}$")
    plt.xlabel("time")
    plt.ylabel(r"$\Phi$")
    plt.title("Effect of homeostatic strength on collective order")
    plt.legend(frameon=False)
    plt.tight_layout()
    fname = os.path.join(outdir, "phi_alpha_comparison.pdf")
    plt.savefig(fname, format="pdf")
    plt.close()
    return fname


def save_phase_heatmap(outdir: str, coupling_range=None, alpha_range=None, rng_seed: int = 1000):
    if coupling_range is None:
        coupling_range = np.linspace(0.02, 0.30, 8)
    if alpha_range is None:
        alpha_range = np.linspace(0.2, 2.5, 8)
    final_Phi = np.zeros((len(coupling_range), len(alpha_range)), dtype=float)

    for i, kappa in enumerate(coupling_range):
        for j, a in enumerate(alpha_range):
            seed = int(rng_seed + i * 100 + j)
            sim = TCHSimulator(N=30, dt=0.05, steps=600,
                               coupling_S=float(kappa), coupling_E=0.03, seed=seed)
            sim.run(alpha_homeo=float(a), eta=0.02)
            final_Phi[i, j] = float(sim.history['Phi'][-1])

    X, Y = np.meshgrid(alpha_range, coupling_range)
    fname = os.path.join(outdir, "phi_phase_heatmap.pdf")
    fig, ax = plt.subplots(figsize=(6.0, 4.8))
    pcm = ax.pcolormesh(X, Y, final_Phi, shading='auto')
    ax.set_xlabel(r"$\alpha_{\mathrm{homeo}}$")
    ax.set_ylabel("coupling_S")
    ax.set_title(r"Final $\Phi$ (steady-state) — parameter sweep", pad=14)
    cbar = fig.colorbar(pcm, ax=ax, pad=0.02)
    cbar.set_label(r"Final $\Phi$", rotation=90, labelpad=15)
    cbar.formatter.set_powerlimits((-2, 2))
    cbar.update_ticks()
    fig.subplots_adjust(left=0.14, right=0.86, top=0.88, bottom=0.12)
    fig.savefig(fname, format="pdf", bbox_inches="tight")
    plt.close(fig)

    heatmap_csv = os.path.join(outdir, "phi_heatmap_matrix.csv")
    df = pd.DataFrame(final_Phi, index=np.round(coupling_range, 6),
                      columns=np.round(alpha_range, 6))
    df.index.name = "coupling_S"
    df.columns.name = "alpha_homeo"
    df.to_csv(heatmap_csv, float_format="%.6f")
    return fname, heatmap_csv, final_Phi, coupling_range, alpha_range


# -----------------------------
# Reproducibility test
# -----------------------------
def test_reproducibility():
    """Verify that fixed seeds produce identical results across runs"""
    print("Testing reproducibility...")
    sim1 = TCHSimulator(seed=42)
    sim1.run(alpha_homeo=1.2)
    result1 = sim1.history['Phi'][-1]

    sim2 = TCHSimulator(seed=42)
    sim2.run(alpha_homeo=1.2)
    result2 = sim2.history['Phi'][-1]

    assert abs(result1 - result2) < 1e-10, f"Reproducibility failed: {result1} vs {result2}"
    print("✓ Reproducibility verified: identical seeds produce identical results")

    sim3 = TCHSimulator(seed=123)
    sim3.run(alpha_homeo=1.2)
    result3 = sim3.history['Phi'][-1]
    assert abs(result1 - result3) > 1e-6, "Random seeds not producing variation"
    print("✓ Random seed variation confirmed")


# -----------------------------
# Main
# -----------------------------
def main():
    outdir = ensure_outdir("figures")
    
    # Generate all figures
    baseline_pdf, baseline_sim = save_baseline_time_series(outdir=outdir, rng_seed=1234)
    alpha_pdf = save_alpha_comparison(outdir=outdir, alphas=(0.5, 1.0, 2.0), rng_seed=42)
    heatmap_pdf, heatmap_csv, final_Phi, coupling_range, alpha_range = save_phase_heatmap(
        outdir=outdir, rng_seed=1000
    )

    # Save summary CSV
    rows = []
    rows.append({
        "experiment": "baseline",
        "coupling_S": 0.15,
        "alpha_homeo": 1.2,
        "final_Phi": float(baseline_sim.history['Phi'][-1]),
        "mean_S": float(baseline_sim.history['mean_S'][-1]),
        "mean_E": float(baseline_sim.history['mean_E'][-1])
    })
    for a in (0.5, 1.0, 2.0):
        s = TCHSimulator(N=30, dt=0.05, steps=600, coupling_S=0.15, coupling_E=0.03, seed=42)
        s.run(alpha_homeo=float(a), eta=0.02)
        rows.append({
            "experiment": f"alpha_{a}",
            "coupling_S": 0.15,
            "alpha_homeo": float(a),
            "final_Phi": float(s.history['Phi'][-1]),
            "mean_S": float(s.history['mean_S'][-1]),
            "mean_E": float(s.history['mean_E'][-1])
        })

    summary_df = pd.DataFrame(rows)
    summary_csv = os.path.join(outdir, "tch_summary_results.csv")
    summary_df.to_csv(summary_csv, index=False, float_format="%.6f")

    print("Files written to:", os.path.abspath(outdir))
    print(" -", os.path.basename(baseline_pdf))
    print(" -", os.path.basename(alpha_pdf))
    print(" -", os.path.basename(heatmap_pdf))
    print(" -", os.path.basename(summary_csv))
    print(" -", os.path.basename(heatmap_csv))


if __name__ == "__main__":
    test_reproducibility()
    main()
