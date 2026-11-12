
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



FC = 3e9
LAM = 0.1
K = 2 * np.pi / LAM                 # wavenumber
C0 = (LAM / (4 * np.pi)) ** 2
A_IRS, A_DIR = 2.2, 3.5             # path-loss exponents
KAPPA = 10 ** 0.3                   # 3 dB Rician factor
P_DBM, N0_DBM = 40.0, -90.0
PBAR = 10 ** ((P_DBM - N0_DBM) / 10)
M_DEF, J_DEF = 4, 3
D_MIN = LAM / 2                     # min MA spacing
D_ELEM = LAM / 2                    # IRS element spacing

# IRS reference points (Fig. 3)
IRS_REF = np.array([[ 5.,   0., 12.],
                    [ 0.,  12.,  5.],
                    [ 0., -12.,  5.],
                    [10.,  25.,  5.],
                    [10., -25.,  5.]])

# target region and 5x5 m areas sampled on a 1 m grid
REG_X, REG_Y, SIDE, STEP = (50., 70.), (-40., 40.), 5.0, 1.0

# Things the paper doesn't specify. My choices, edit here.
ASSUMPTIONS = dict(
    irs_layout="rotated",
    # (theta, phi) per panel in deg. theta=90 -> horizontal (IRS 1),
    # theta=0 -> vertical wall panel. phi rotates the in-plane x axis.
    panel_angles=[(90., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
    steering="farfield",      # or "exact" (spherical per-element phase)
    irs_subset="first_L",     # for L < 5 just use the first L IRSs
    grid_points=36,           # 6x6 points per area
    trials=10,
    ao_iters=180,
    ao_restarts=3,
    fpa_axis="y",             # FPA benchmark ULA along y
    # Direct link: Eq. (5) as printed gives a 27-35 dB floor that's above
    # everything in Figs 4-8 and can't be optimised. Paper says the LoS is
    # blocked, so take f_u = 0.
    include_direct=False,
)


# ---- geometry --------------------------------------------------------------
def upa_shape(n):
    n1 = max(1, int(round(np.sqrt(n))))
    return n1, int(np.ceil(n / n1))


def panel_rotation(theta_deg, phi_deg):
    # columns: in-plane axes u1, u2 and the normal
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    u1 = np.array([np.cos(ph), np.sin(ph), 0.])
    z = np.array([0., 0., 1.])
    u2v = np.cross(u1, z)
    u2v /= np.linalg.norm(u2v)
    u2 = np.cos(th) * z + np.sin(th) * u2v
    u2 /= np.linalg.norm(u2)
    n = np.cross(u1, u2)
    n /= np.linalg.norm(n)
    return np.column_stack([u1, u2, n])


def irs_elements(l, n_elem):
    # panel centred on the reference point; element 0 is used as p_{l,1}.
    # (the shift vs. a corner reference is <0.2 m, negligible)
    n1, n2 = upa_shape(n_elem)
    R = panel_rotation(*ASSUMPTIONS["panel_angles"][l])
    i, q = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    loc = np.stack([i.ravel(), q.ravel()], 1).astype(float) * D_ELEM
    loc -= loc.mean(0)
    loc = np.column_stack([loc, np.zeros(len(loc))])
    return IRS_REF[l] + loc[:n_elem] @ R.T


def sample_areas(J, rng):
    out = []
    while len(out) < J:
        cx = rng.uniform(REG_X[0], REG_X[1] - SIDE)
        cy = rng.uniform(REG_Y[0], REG_Y[1] - SIDE)
        if all(abs(cx - a) >= SIDE or abs(cy - b) >= SIDE for a, b in out):
            out.append((cx, cy))
    return out


# ---- channel model, Eqs (1)-(5), and Prop. 1 -------------------------------
class System:
    def __init__(self, corners, L, n_elem, M, include_direct=None):
        self.L, self.M, self.ne, self.N = L, M, n_elem, L * n_elem
        self.J = len(corners)
        if include_direct is None:
            include_direct = ASSUMPTIONS["include_direct"]

        # candidate user locations
        s = np.arange(0., SIDE + 1e-9, STEP)
        gx, gy = np.meshgrid(s, s, indexing="ij")
        U, jj = [], []
        for j, (cx, cy) in enumerate(corners):
            U.append(np.stack([cx + gx.ravel(), cy + gy.ravel(),
                               np.zeros(gx.size)], 1))
            jj.append(np.full(gx.size, j))
        self.U, self.jidx = np.concatenate(U), np.concatenate(jj)
        self.G = len(self.U)
        self.gj = [np.where(self.jidx == j)[0] for j in range(self.J)]

        P1 = IRS_REF[:L]
        self.d_l = np.linalg.norm(P1, axis=1)
        self.a_t = P1 / self.d_l[:, None]            # BS -> IRS directions
        # alpha, omega (26) and psi (27)
        self.al = self.a_t[:, 1][:, None] - self.a_t[:, 1][None, :]
        self.om = self.a_t[:, 2][:, None] - self.a_t[:, 2][None, :]
        self.psi = K ** 2 * (self.al ** 2 + self.om ** 2)

        # e_l, Eq. (2)
        self.Pel = [irs_elements(l, n_elem) for l in range(L)]
        self.e = np.stack([np.exp(-1j * K * (self.Pel[l] - self.Pel[l][0])
                                  @ self.a_t[l]) for l in range(L)])

        # IRS -> user, Eqs (3),(4)
        diff = self.U[None] - P1[:, None]
        self.d_lj = np.linalg.norm(diff, axis=2)
        a_lj = diff / self.d_lj[..., None]
        if ASSUMPTIONS["steering"] == "exact":
            hb = []
            for l in range(L):
                dn = np.linalg.norm(self.U[:, None, :] - self.Pel[l][None], axis=2)
                hb.append(np.exp(-1j * K * (dn - dn[:, [0]])))
            self.hbarH = np.stack(hb)
        else:
            self.hbarH = np.stack([
                np.exp(1j * K * (a_lj[l] @ (self.Pel[l] - self.Pel[l][0]).T))
                for l in range(L)])

        self.X = self.hbarH * self.e[:, None, :]
        self.Xf = self.X.transpose(1, 0, 2).reshape(self.G, self.N)
        self.Xfc = np.conj(self.Xf)

        rho = np.sqrt(C0 * self.d_l ** -A_IRS * KAPPA / (KAPPA + 1))
        sig = np.sqrt(C0 * self.d_lj ** -A_IRS * KAPPA / (KAPPA + 1))
        self.w = (rho[:, None] * sig).T              # (G, L)

        # constant part of (11). Lambda really depends on l too.
        Lam = (C0 ** 2 * self.d_lj ** -A_IRS
               * (self.d_l ** -A_IRS)[:, None] / (KAPPA + 1) ** 2)
        scat = ((2 * KAPPA + 1) * Lam * M * n_elem).sum(0)
        du = np.linalg.norm(self.U, axis=1)
        if include_direct:
            self.C_dir = PBAR * C0 * du ** -A_DIR * M
        else:
            self.C_dir = np.zeros(self.G)
        self.C_sca = PBAR * scat
        self.C_u = self.C_sca + self.C_dir

    def gH(self, T):
        # T is (M, 2) with columns (y, z)
        return np.exp(1j * K * (T[:, 0] * self.a_t[:, 1][:, None]
                                + T[:, 1] * self.a_t[:, 2][:, None]))

    def c(self, v):
        return (self.Xf * v).reshape(self.G, self.L, self.ne).sum(2)

    def snr(self, T, v, sel=None):
        if sel is None:
            sel = np.arange(self.G)
        r = (self.w[sel] * self.c(v)[sel]) @ self.gH(T)
        return PBAR * np.einsum("gm,gm->g", r, np.conj(r)).real + self.C_u[sel]


# ---- Monte Carlo check of (11) ---------------------------------------------
def mc_snr(S, T, v, u, n_mc, rng):
    gH, tot = S.gH(T), 0.0
    for _ in range(n_mc):
        acc = np.zeros(S.M, complex)
        for l in range(S.L):
            Gt = (rng.standard_normal((S.ne, S.M))
                  + 1j * rng.standard_normal((S.ne, S.M))) / np.sqrt(2)
            Gl = np.sqrt(C0 * S.d_l[l] ** -A_IRS) * (
                np.sqrt(KAPPA / (KAPPA + 1)) * np.outer(S.e[l], gH[l])
                + np.sqrt(1 / (KAPPA + 1)) * Gt)
            ht = (rng.standard_normal(S.ne)
                  + 1j * rng.standard_normal(S.ne)) / np.sqrt(2)
            hH = np.sqrt(C0 * S.d_lj[l, u] ** -A_IRS) * (
                np.sqrt(KAPPA / (KAPPA + 1)) * S.hbarH[l, u]
                + np.sqrt(1 / (KAPPA + 1)) * ht)
            acc += (hH * v[l * S.ne:(l + 1) * S.ne]) @ Gl
        f = np.sqrt(S.C_dir[u] / (PBAR * S.M)) * (
            rng.standard_normal(S.M) + 1j * rng.standard_normal(S.M)) / np.sqrt(2)
        acc += f
        tot += PBAR * np.vdot(acc, acc).real
    return tot / n_mc


# ---- Algorithm 1 -----------------------------------------------------------
# MA update: surrogate (23)/(24) with curvature bound (27)
# IRS update: lower bound (35)
def _temp(it, ni):
    # softmin temperature, annealed 0.5 -> 0.02
    return 0.5 * (0.02 / 0.5) ** (it / (ni - 1))


def _weights(h, theta):
    sp = h.max() - h.min()
    if sp <= 0 or theta <= 0:
        w = np.zeros_like(h)
        w[np.argmin(h)] = 1.0
        return w
    z = -(h - h.min()) / (theta * sp)
    w = np.exp(z - z.max())
    return w / w.sum()


def _spacing(T, A):
    # box (7c) + min spacing (7d), crude pairwise push
    T = np.clip(T, -A / 2, A / 2)
    for _ in range(60):
        d = np.linalg.norm(T[:, None] - T[None], axis=2)
        np.fill_diagonal(d, np.inf)
        if d.min() >= D_MIN - 1e-12:
            break
        for m, q in np.argwhere(d < D_MIN):
            if m >= q:
                continue
            vec = T[m] - T[q]
            r = np.linalg.norm(vec)
            if r < 1e-9:
                vec, r = np.array([1., 0.]) * 1e-3, 1e-3
            T[m] += (D_MIN - r) / 2 * 1.02 * vec / r
            T[q] -= (D_MIN - r) / 2 * 1.02 * vec / r
        T = np.clip(T, -A / 2, A / 2)
    return T


def _irs_step(S, T, v, om, sel, esel):
    r = (S.w * S.c(v)) @ S.gH(T)
    zeta = S.w * (r @ np.conj(S.gH(T)).T)
    zf = np.repeat(zeta, S.ne, 1)
    g = (om[sel, None] * zf[np.ix_(sel, esel)] * S.Xfc[np.ix_(sel, esel)]).sum(0)
    out = v.copy()
    out[esel] = np.where(np.abs(g) > 0, g / np.maximum(np.abs(g), 1e-300), v[esel])
    return out


def _beta(S, v, sel, om):
    # Re{b_ll'} from (19), weighted
    wc = S.w * S.c(v)
    return PBAR * np.einsum("g,gl,gm->lm", om[sel], wc[sel],
                            np.conj(wc[sel])).real


def _ma_step(S, T, beta, A):
    Sc = (np.abs(beta) * S.psi).sum()
    if Sc <= 0:
        return T
    nu = K * (T[:, 0][:, None, None] * S.al + T[:, 1][:, None, None] * S.om)
    sn = np.sin(nu)
    gy = -K * np.einsum("lm,klm,lm->k", beta, sn, S.al)
    gz = -K * np.einsum("lm,klm,lm->k", beta, sn, S.om)
    T = np.clip(T + np.stack([gy, gz], 1) / Sc, -A / 2, A / 2)
    return _spacing(T, A)


def fpa(M):
    p = (np.arange(M) - (M - 1) / 2) * (LAM / 2)
    if ASSUMPTIONS["fpa_axis"] == "y":
        return np.stack([p, np.zeros(M)], 1)
    return np.stack([np.zeros(M), p], 1)


def _init_T(M, A, rng, mode):
    if mode == "ula":
        T = np.stack([(np.arange(M) - (M - 1) / 2) * (LAM / 2), np.zeros(M)], 1)
    else:
        T = rng.uniform(-A / 2, A / 2, (M, 2))
    return _spacing(np.clip(T.astype(float), -A / 2, A / 2), A)


def _init_v(S, sel):
    x = S.Xf[sel[len(sel) // 2]]
    return np.conj(x) / np.abs(x)


def solve(S, scheme, A, warm=None, seed=0):
    # schemes: ma_irs, ma_stairs, shared_ma_stairs, fpa_irs, fpa_stairs
    rng = np.random.default_rng(seed)
    ni, nr = ASSUMPTIONS["ao_iters"], ASSUMPTIONS["ao_restarts"]
    ma_ad = scheme in ("ma_irs", "ma_stairs")
    irs_ad = scheme in ("ma_irs", "fpa_irs")
    is_fpa = scheme.startswith("fpa")
    allE = np.arange(S.N)

    # per-area IRS -> problem splits across areas
    if irs_ad:
        vals, Ts, vs = [], [], []
        n_runs = 1 if is_fpa else nr
        for j in range(S.J):
            sel = S.gj[j]
            best = (-np.inf, None, None)
            wj = None
            if warm is not None:
                wj = (warm[0][j],
                      warm[1][j] if isinstance(warm[1], list) else warm[1])
            for r in range(n_runs + (0 if wj is None else 1)):
                if wj is not None and r == n_runs:
                    T = fpa(S.M) if is_fpa else _spacing(
                        np.clip(wj[0].copy(), -A / 2, A / 2), A)
                    v = wj[1].copy()
                else:
                    T = fpa(S.M) if is_fpa else _init_T(
                        S.M, A, rng, "ula" if r == 0 else "rand")
                    v = _init_v(S, sel)
                cur = (-np.inf, None, None)
                for it in range(ni):
                    h = S.snr(T, v, sel)
                    om = np.zeros(S.G)
                    om[sel] = _weights(h, _temp(it, ni))
                    if h.min() > cur[0]:
                        cur = (h.min(), T.copy(), v.copy())
                    v = _irs_step(S, T, v, om, sel, allE)
                    if not is_fpa:
                        T = _ma_step(S, T, _beta(S, v, sel, om), A)
                h = S.snr(T, v, sel)
                if h.min() > cur[0]:
                    cur = (h.min(), T, v)
                if cur[0] > best[0]:
                    best = cur
            vals.append(best[0])
            Ts.append(best[1])
            vs.append(best[2])
        return float(min(vals)), Ts, vs

    # shared IRS phases: P2, P3, FPA-staIRS
    best = (-np.inf, None, None)
    for r in range(nr + (0 if warm is None else 1)):
        if warm is not None and r == nr:
            Tw, vw = warm
            T0 = Tw[0] if isinstance(Tw, list) else Tw
            if is_fpa:
                Tl = [fpa(S.M)] * S.J
            else:
                Tl = [_spacing(np.clip(T0.copy(), -A / 2, A / 2), A)] * S.J
            v = (vw[0] if isinstance(vw, list) else vw).copy()
        else:
            mode = "ula" if r == 0 else "rand"
            if is_fpa:
                Tl = [fpa(S.M)] * S.J
            elif ma_ad:
                Tl = [_init_T(S.M, A, rng, mode) for _ in range(S.J)]
            else:
                Tl = [_init_T(S.M, A, rng, mode)] * S.J
            v = _init_v(S, np.arange(S.G))

        cur = (-np.inf, None, None)
        for it in range(ni):
            h = np.empty(S.G)
            for j in range(S.J):
                h[S.gj[j]] = S.snr(Tl[j], v, S.gj[j])
            om = _weights(h, _temp(it, ni))
            if h.min() > cur[0]:
                cur = (h.min(), [t.copy() for t in Tl], v.copy())

            # Theta is shared so the IRS step sees every location
            zeta = np.empty((S.G, S.L), complex)
            cc = S.c(v)
            for j in range(S.J):
                sel, g = S.gj[j], S.gH(Tl[j])
                rr = (S.w[sel] * cc[sel]) @ g
                zeta[sel] = S.w[sel] * (rr @ np.conj(g).T)
            zf = np.repeat(zeta, S.ne, 1)
            gr = (om[:, None] * zf * S.Xfc).sum(0)
            v = np.where(np.abs(gr) > 0, gr / np.maximum(np.abs(gr), 1e-300), v)

            if not is_fpa:
                if ma_ad:
                    Tl = [_ma_step(S, Tl[j], _beta(S, v, S.gj[j], om), A)
                          for j in range(S.J)]
                else:
                    T_new = _ma_step(S, Tl[0], _beta(S, v, np.arange(S.G), om), A)
                    Tl = [T_new] * S.J
        if cur[0] > best[0]:
            best = cur
        if is_fpa and warm is None:
            break
    return float(best[0]), best[1], best[2]


SCHEMES = ("ma_irs", "ma_stairs", "shared_ma_stairs", "fpa_irs", "fpa_stairs")


def solve_all(S, A, seed=0):
    # go from most to least constrained and warm-start each from the
    # previous, so the ordering implied by nesting actually holds
    o = {}
    o["fpa_stairs"], T, v = solve(S, "fpa_stairs", A, seed=seed)
    o["fpa_irs"], _, _ = solve(S, "fpa_irs", A, warm=(T, v), seed=seed)
    o["shared_ma_stairs"], T3, v3 = solve(S, "shared_ma_stairs", A,
                                          warm=(T, v), seed=seed)
    o["ma_stairs"], T4, v4 = solve(S, "ma_stairs", A, warm=(T3, v3), seed=seed)
    o["ma_irs"], _, _ = solve(S, "ma_irs", A, warm=(T4, v4), seed=seed)
    return o


# ---- plotting --------------------------------------------------------------
os.makedirs("figs2", exist_ok=True)
os.makedirs("cache2", exist_ok=True)

MODES = [True, False]   # include_direct for left / right panel
ST = {
    "ma_irs":           dict(label="Area-adaptive MA-IRS",    c="#1f4fd8", m="*", ls="-"),
    "ma_stairs":        dict(label="Area-adaptive MA-staIRS", c="#e21ce2", m="v", ls="--"),
    "shared_ma_stairs": dict(label="Shared MA-staIRS",        c="#e01b24", m="o", ls="-"),
    "fpa_irs":          dict(label="FPA-(area-adaptive IRS)", c="#1a9c3c", m="D", ls="-"),
    "fpa_stairs":       dict(label="FPA-staIRS",              c="k",       m="",  ls=":"),
}
ORD = list(ST)
OPEN_FIGS = False


def dB(x):
    return 10 * np.log10(x)


def save_fig(fig, name):
    p = os.path.abspath(os.path.join("figs2", name + ".png"))
    fig.savefig(p, dpi=185, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", p, flush=True)
    if OPEN_FIGS == "vscode":
        import shutil, subprocess
        exe = shutil.which("code") or shutil.which("code-insiders")
        if exe:
            subprocess.run([exe, "-r", p], check=False)
        else:
            print("  (code CLI not on PATH)")
    elif OPEN_FIGS:
        import webbrowser
        webbrowser.open("file://" + p)


def run(tag, xs, build, trials, inc, single=False):
    # same area layouts across x values and across both panels
    f = f"cache2/{tag}_{'dir' if inc else 'nodir'}_t{trials}.npz"
    if os.path.exists(f):
        return dict(np.load(f, allow_pickle=True))
    keys = ["ma_irs"] if single else ORD
    acc = {k: np.zeros((len(xs), trials)) for k in keys}
    ASSUMPTIONS["include_direct"] = inc
    for t in range(trials):
        for i, x in enumerate(xs):
            J, L, ne, M, A = build(x)
            S = System(sample_areas(J, np.random.default_rng(10_000 + t)), L, ne, M)
            if single:
                acc["ma_irs"][i, t] = solve(S, "ma_irs", A, seed=t)[0]
            else:
                r = solve_all(S, A, seed=t)
                for k in keys:
                    acc[k][i, t] = r[k]
        print(f"    {tag} direct={inc} trial {t+1}/{trials}", flush=True)
    out = {k: dB(v.mean(1)) for k, v in acc.items()}
    np.savez(f, **out)
    return out


def panel(ax, xs, d, xlabel, title, keys=None):
    for s in (keys or ORD):
        st = ST[s]
        ax.plot(xs, d[s], color=st["c"], marker=st["m"], ls=st["ls"],
                ms=8 if st["m"] == "*" else 5.5, lw=1.5, mfc="none",
                label=st["label"])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=9.5)
    ax.grid(True, ls=":", lw=.6, alpha=.7)
    ax.tick_params(labelsize=8.5)
    ax.set_xticks(xs)


def two_panel(name, xs, build, trials, xlabel, single=False):
    ds = [run(name, xs, build, trials, b, single) for b in MODES]
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.9))
    titles = ["(a) Eq. (5) retained: direct link included",
              "(b) $\\mathbf{f}_{u_j}=0$  (reproduces Figs. 4-8)"]
    for ax, d, ttl in zip(axs, ds, titles):
        panel(ax, xs, d, xlabel, ttl, ["ma_irs"] if single else None)
    axs[0].set_ylabel("Average worst-case SNR (dB)", fontsize=10)
    axs[0].legend(fontsize=7.5, framealpha=.95)
    lo, hi = axs[0].get_ylim()
    if hi - lo < 4:   # left panel is squashed flat otherwise
        mid = (lo + hi) / 2
        axs[0].set_ylim(mid - 2.5, mid + 2.5)
    save_fig(fig, name)
    return ds


def fig_setup():
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.add_patch(plt.Rectangle((REG_X[0], REG_Y[0]), REG_X[1] - REG_X[0],
                               REG_Y[1] - REG_Y[0], fc="0.92", ec="0.6",
                               ls="--", label="target-area region"))
    for c in sample_areas(3, np.random.default_rng(10_000)):
        ax.add_patch(plt.Rectangle(c, SIDE, SIDE, fc="#f2c14e", ec="k", lw=.8))
    ax.plot(0, 0, "s", ms=11, color="#3a7d44", label="BS")
    for l in range(5):
        P = irs_elements(l, 20)
        ax.plot(P[:, 0], P[:, 1], "-", lw=4, color="#1f4fd8", alpha=.9,
                solid_capstyle="butt")
        ax.annotate(f"IRS {l+1}", IRS_REF[l][:2], textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.plot([], [], "-", lw=4, color="#1f4fd8", label="IRS aperture (top view)")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Simulation setup, top view (cf. paper Fig. 3)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, ls=":", lw=.6)
    ax.set_aspect("equal")
    ax.set_xlim(-8, 78)
    ax.set_ylim(-46, 46)
    save_fig(fig, "fig3_setup")


def fig_convergence(seed=0):
    S = System(sample_areas(3, np.random.default_rng(10_000)), 3, 20, 4)
    A, ni = 5 * LAM, ASSUMPTIONS["ao_iters"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for sc in ORD:
        rng = np.random.default_rng(seed)
        is_f = sc.startswith("fpa")
        ma_ad = sc in ("ma_irs", "ma_stairs")
        irs_ad = sc in ("ma_irs", "fpa_irs")
        Tl = [fpa(S.M)] * S.J if is_f else [_init_T(S.M, A, rng, "ula")] * S.J
        if irs_ad:
            V = [_init_v(S, S.gj[j]) for j in range(S.J)]
        else:
            V = [_init_v(S, np.arange(S.G))] * S.J
        best, tr = -np.inf, []
        for it in range(ni):
            h = np.concatenate([S.snr(Tl[j], V[j], S.gj[j]) for j in range(S.J)])
            best = max(best, h.min())
            tr.append(best)
            om = _weights(h, _temp(it, ni))
            for j in range(S.J):
                sel = S.gj[j]
                oj = np.zeros(S.G)
                oj[sel] = om[sel]
                V[j] = _irs_step(S, Tl[j], V[j], oj, sel, np.arange(S.N))
            if not irs_ad:
                V = [V[0]] * S.J
            if not is_f:
                if ma_ad:
                    Tl = [_ma_step(S, Tl[j], _beta(S, V[j], S.gj[j], om), A)
                          for j in range(S.J)]
                else:
                    T_new = _ma_step(S, Tl[0], _beta(S, V[0], np.arange(S.G), om), A)
                    Tl = [T_new] * S.J
        ax.plot(dB(np.array(tr)), color=ST[sc]["c"], ls=ST[sc]["ls"], lw=1.5,
                label=ST[sc]["label"])
    ax.set_xlabel("AO iteration $r$")
    ax.set_ylabel("Best-so-far worst-case SNR (dB)")
    ax.set_title("Algorithm 1 convergence ($J=3$, $L=3$, $N_e=20$, $M=4$)\n"
                 "single restart, no warm start (final results use %d restarts\n"
                 "+ warm starts, so they end up higher)"
                 % ASSUMPTIONS["ao_restarts"], fontsize=8.5)
    ax.grid(True, ls=":", lw=.6)
    ax.legend(fontsize=7.5, loc="lower right")
    save_fig(fig, "figS1_convergence")


def fig_selectivity():
    # how well a fixed IRS can tell the two extreme users apart, vs panel azimuth
    uA, uB = np.array([60., -40., 0.]), np.array([60., 40., 0.])
    phis = np.arange(0, 91, 5.0)
    saved = list(ASSUMPTIONS["panel_angles"])
    corr, spread = [], []
    for ph in phis:
        ASSUMPTIONS["panel_angles"] = [(90., 0.), (0., -ph), (0., ph),
                                       (0., -ph), (0., ph)]
        pa, pb = [], []
        for l in range(3):
            P = irs_elements(l, 20)
            for u, acc in ((uA, pa), (uB, pb)):
                a = u - P[0]
                a /= np.linalg.norm(a)
                acc.append(K * ((P - P[0]) @ a))
        pa, pb = np.concatenate(pa), np.concatenate(pb)
        d = np.angle(np.exp(1j * (pa - pb)))
        corr.append(np.abs(np.vdot(np.exp(1j * pa), np.exp(1j * pb))) / 60)
        spread.append(d.max() - d.min())
    ASSUMPTIONS["panel_angles"] = saved

    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(phis, corr, "o-", color="#1f4fd8", ms=4,
            label=r"$|\bar{\mathbf{h}}(u_A)^H\bar{\mathbf{h}}(u_B)|/N$")
    ax.set_xlabel(r"vertical-panel azimuth $|\phi_\ell|$ (deg)")
    ax.set_ylabel("steering-vector correlation")
    ax2 = ax.twinx()
    ax2.plot(phis, spread, "s--", color="#e01b24", ms=4,
             label="differential phase spread")
    ax2.set_ylabel("differential phase spread (rad)", color="#e01b24")
    ax.set_title("IRS phase selectivity vs panel orientation\n"
                 "(lower correlation -> more gain from adaptive IRS)", fontsize=9)
    ax.grid(True, ls=":", lw=.6)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    save_fig(fig, "figS2_selectivity")


def fig8(T):
    # MA-IRS only, A = 8 lambda, L = 2, Ne = N/L, Ctot = rho*ce*M + ce*N
    configs = [("a", [(5, 120), (10, 120), (20, 120), (30, 120)],
                "$C_{\\mathrm{tot}}=120c_{\\mathrm{e}}$"),
               ("b", [(20, 120), (20, 240), (20, 360)], "$\\rho=20$")]

    for inc in MODES:
        ASSUMPTIONS["include_direct"] = inc
        for lbl, pairs, _ in configs:
            fn = f"cache2/fig8{lbl}_{'dir' if inc else 'nodir'}_t{T}.npz"
            if os.path.exists(fn):
                continue
            out = {}
            for rho, ct in pairs:
                Ms, vals = [], []
                M = 1
                while ct - rho * M >= 6:
                    ne = int((ct - rho * M) // 2)
                    acc = np.zeros(T)
                    for t in range(T):
                        S = System(sample_areas(3, np.random.default_rng(20_000 + t)),
                                   2, ne, M)
                        acc[t] = solve(S, "ma_irs", 8 * LAM, seed=t)[0]
                    Ms.append(M)
                    vals.append(acc.mean())
                    M += 1
                out[f"M_{rho}_{ct}"] = np.array(Ms)
                out[f"v_{rho}_{ct}"] = dB(np.array(vals))
                print(f"    fig8{lbl} direct={inc} rho={rho} Ctot={ct} done", flush=True)
            np.savez(fn, **out)

    for lbl, pairs, ttl in configs:
        fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.9))
        for ax, inc, sub in zip(axs, MODES, ["(a) direct link included",
                                              "(b) $\\mathbf{f}_{u_j}=0$"]):
            d = dict(np.load(f"cache2/fig8{lbl}_{'dir' if inc else 'nodir'}_t{T}.npz"))
            for rho, ct in pairs:
                M, v = d[f"M_{rho}_{ct}"], d[f"v_{rho}_{ct}"]
                if lbl == "a":
                    lab = f"$\\rho={rho}$"
                else:
                    lab = f"$C_{{\\mathrm{{tot}}}}={ct}c_{{\\mathrm{{e}}}}$"
                ax.plot(M, v, marker="o", ms=4.5, lw=1.4, mfc="none", label=lab)
                i = int(np.argmax(v))
                ax.annotate(f"$({M[i]},{ct - rho * M[i]})$", xy=(M[i], v[i]),
                            xytext=(M[i] + 1, v[i] - 1.6), fontsize=7,
                            arrowprops=dict(arrowstyle="->", lw=.6))
            ax.set_xlabel("Number of transmit MAs, $M$", fontsize=10)
            ax.set_title(f"{sub} -- {ttl}", fontsize=9.5)
            ax.grid(True, ls=":", lw=.6, alpha=.7)
            ax.legend(fontsize=7.5)
        axs[0].set_ylabel("Average worst-case SNR (dB)", fontsize=10)
        save_fig(fig, f"fig8{lbl}_cost")


def check():
    rng = np.random.default_rng(0)
    print("Prop. 1 (Eq. 11) vs Monte Carlo")
    for L in (1, 3):
        S = System(sample_areas(3, np.random.default_rng(7)), L, 12, 4)
        Tm = _init_T(4, 0.5, rng, "rand")
        v = np.exp(1j * rng.uniform(0, 2 * np.pi, S.N))
        for u in (5, 40):
            an = S.snr(Tm, v)[u]
            mc = mc_snr(S, Tm, v, u, 40000, np.random.default_rng(3))
            print("  L=%d u=%-3d analytic %10.3f  MC %10.3f  err %+.4f dB"
                  % (L, u, an, mc, 10 * np.log10(mc / an)))

    areas = sample_areas(3, np.random.default_rng(7))
    S_dir = System(areas, 3, 20, 4, include_direct=True)
    S_nodir = System(areas, 3, 20, 4, include_direct=False)
    print("\nConstant floor C_u with Eq. (5) as printed:")
    print("  direct link : %6.2f .. %6.2f dB" % (dB(S_dir.C_dir.min()), dB(S_dir.C_dir.max())))
    print("  scattering  : %6.2f .. %6.2f dB" % (dB(S_dir.C_sca.min()), dB(S_dir.C_sca.max())))
    print("  -> above everything in Fig. 4 (13.8-19.3 dB)")
    print("  with f_u = 0: %6.2f .. %6.2f dB" % (dB(S_nodir.C_u.min()), dB(S_nodir.C_u.max())))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="4",
                    choices=["3", "4", "5", "6", "7", "8", "conv", "sel", "all"])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--open", action="store_true", dest="open_figs",
                    help="open figures in the default image viewer")
    ap.add_argument("--vscode", action="store_true", help="open figures in VS Code")
    ap.add_argument("--check", action="store_true", help="Prop. 1 vs Monte Carlo")
    args = ap.parse_args()
    T = args.trials
    OPEN_FIGS = "vscode" if args.vscode else args.open_figs
    print("figures ->", os.path.abspath("figs2"))
    print("cache   ->", os.path.abspath("cache2"), "\n")

    if args.check:
        check()
        raise SystemExit

    f = args.fig
    if f in ("3", "all"):
        fig_setup()
    if f in ("conv", "all"):
        fig_convergence()
    if f in ("sel", "all"):
        fig_selectivity()
    if f in ("4", "all"):
        two_panel("fig4_vs_J", np.arange(1, 7),
                  lambda J: (int(J), 3, 20, 4, 5 * LAM), T,
                  "Number of target areas, $J$")
    if f in ("5", "all"):
        two_panel("fig5_vs_M", np.array([2, 4, 6, 8, 10, 12]),
                  lambda M: (3, 3, 20, int(M), 8 * LAM), T,
                  "Number of transmit MAs, $M$")
    if f in ("6", "all"):
        two_panel("fig6_vs_Ne", np.array([10, 20, 30, 40, 50, 60]),
                  lambda n: (3, 3, int(n), 4, 5 * LAM), T,
                  "Number of elements at each IRS, $N_{\\mathrm{e}}$")
    if f in ("7", "all"):
        two_panel("fig7a_vs_L_Ne20", np.arange(1, 6),
                  lambda L: (3, int(L), 20, 4, 5 * LAM), T,
                  "Number of IRSs, $L$")
        two_panel("fig7b_vs_L_N120", np.arange(1, 6),
                  lambda L: (3, int(L), 120 // int(L), 4, 5 * LAM), T,
                  "Number of IRSs, $L$")
    if f in ("8", "all"):
        fig8(T)
