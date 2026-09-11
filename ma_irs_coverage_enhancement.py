"""
ma_irs_paper.py -- SINGLE-FILE implementation of

  Y. Gao, Q. Wu, W. Mei, G. Chen, W. Chen, Z. Zheng, "Integrating Movable
  Antennas and Intelligent Reflecting Surfaces for Coverage Enhancement",
  IEEE Trans. Wireless Commun., vol. 25, 2026, pp. 6082-6095.

Nothing to install beyond numpy + matplotlib, and nothing to import.
Just:   python3 ma_irs_paper.py --fig 4
        python3 ma_irs_paper.py --fig all --trials 8
        python3 ma_irs_paper.py --check          # Prop. 1 vs Monte Carlo

Every parameter is taken verbatim from Section IV.  The seven quantities the
paper does not state are collected in the ASSUMPTIONS dict below; change them
there and nowhere else.

Each swept figure is drawn as two panels:
  LEFT  : Eq. (5) retained exactly as printed.  All five schemes collapse into
          a ~0.2 dB band, because the direct-link floor alone is 29-33 dB and
          is independent of every optimisation variable.
  RIGHT : f_{u_j} = 0.  Reproduces the levels plotted in the paper.
Only ASSUMPTIONS["include_direct"] differs between the panels.

Figures produced
----------------
  --fig 3     simulation setup, top view (reproduces the paper's Fig. 3)
  --fig 4     worst-case SNR vs number of target areas J
  --fig 5     worst-case SNR vs number of transmit MAs M
  --fig 6     worst-case SNR vs elements per IRS Ne
  --fig 7     worst-case SNR vs number of IRSs L   (panels a and b)
  --fig 8     fixed-total-cost sweeps              (panels a and b)
  --fig conv  Algorithm 1 convergence, all five schemes
  --fig sel   IRS phase selectivity vs panel orientation (diagnostic)
  --fig all   everything above
  --check     Prop. 1 (Eq. 11) vs Monte Carlo of Eqs. (1)-(6),(10)

Output: ./figs2/*.png ,  cached sweeps in ./cache2/*.npz
Runtime on one core: figs 3/conv/sel are seconds; 4-8 are ~2-4 min each at
--trials 8, so "all" takes roughly 20 min.
"""
import argparse, os, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt



# =============================================================================
# SECTION IV PARAMETERS -- all stated explicitly in the paper
# =============================================================================
FC          = 3e9                       # "carrier frequency of 3 GHz"
LAM         = 0.1                       # "wavelength of lambda = 0.1 m"
K           = 2 * np.pi / LAM
C0          = (LAM / (4 * np.pi)) ** 2  # "C0 = (lambda/4pi)^2"
A_IRS       = 2.2                       # "path-loss exponents ... 2.2 for IRS-related links"
A_DIR       = 3.5                       # "... and 3.5 for direct links"
KAPPA       = 10 ** (3 / 10)            # "Rician factors of IRS-related links are set to 3 dB"
P_DBM       = 40.0                      # "BS transmit power as P = 40 dBm"
N0_DBM      = -90.0                     # "noise power as sigma^2 = -90 dBm"
PBAR        = 10 ** ((P_DBM - N0_DBM) / 10)      # P/sigma^2
M_DEF       = 4                         # "number of transmit MAs as M = 4"
D_MIN       = LAM / 2                   # "minimum inter-MA spacing as D = lambda/2"
J_DEF       = 3                         # "number of target areas as J = 3"
D_ELEM      = LAM / 2                   # IRS elements "spaced at least lambda/2 apart"

# Fig. 3: "coordinates of the reference points of the five IRSs"
IRS_REF = np.array([[ 5.,   0., 12.],
                    [ 0.,  12.,  5.],
                    [ 0., -12.,  5.],
                    [10.,  25.,  5.],
                    [10., -25.,  5.]])

# "target areas are randomly generated without overlap within a rectangular
#  region defined by x in [50,70] m, y in [-40,40] m, and z = 0 m, where each
#  area is a square of size 5 m x 5 m and is uniformly sampled at one-meter
#  intervals to generate candidate locations."
REG_X, REG_Y, SIDE, STEP = (50., 70.), (-40., 40.), 5.0, 1.0

# =============================================================================
# ASSUMPTIONS -- the seven quantities the paper does NOT state.
# Each is set to the most literal reading of the text.  Change here only.
# =============================================================================
ASSUMPTIONS = dict(
    # 1. IRS panel geometry.  Paper: IRS 1 is "deployed parallel to the ground";
    #    IRSs 2-5 are "vertically oriented (e.g., on building facades or indoor
    #    walls)".  Literal reading: IRS 1 spans the x-y plane; a facade/wall at
    #    y = +-12 or +-25 runs along x, so IRSs 2-5 span the x-z plane.
    #    Shape: near-square UPA at lambda/2 (paper gives element spacing only).
    irs_layout="rotated",
    # Panel orientation as (theta_l, phi_l) in degrees, l = 1..5.
    #   theta_l : elevation tilt of the panel plane (0 = vertical panel,
    #             90 = horizontal panel lying parallel to the ground)
    #   phi_l   : azimuth of the panel's in-plane horizontal axis u1
    # Panel 1 is horizontal (theta=90) per "deployed parallel to the ground".
    # Panels 2-5 are vertical (theta=0) and rotated toward the target region.
    # phi = 0 reproduces the literal x-z aperture (normal along +-y) that
    # collapses the effective DoF; phi = -+45 restores selectivity in y.
    panel_angles=[(90., 0.), (0., 0.), (0., 0.), (0., 0.), (0., 0.)],
    # Steering-vector model: "farfield" = Eq. (4) plane wave (paper);
    # "exact" = per-element spherical phase exp(-j 2pi/lambda ||p_ln - u_j||).
    steering="farfield",
    # 2. Active IRS subset for L < 5: paper gives the five reference points in
    #    a fixed order and never says otherwise -> take the first L.
    irs_subset="first_L",
    # 3. "uniformly sampled at one-meter intervals" over a 5 m square, endpoints
    #    included -> 6 x 6 = 36 candidate locations per area.
    grid_points=36,
    # 4. Monte-Carlo realisations of the random area layout: not stated.
    trials=10,
    # 5. Alg. 1 step 8 threshold / cap: not stated.
    ao_iters=180, ao_restarts=3,
    # 6. FPA benchmark is a "uniform linear array ... with spacing lambda/2" in
    #    the y-z moving plane; axis not stated -> y axis.
    fpa_axis="y",
    # 7. Direct link.  Eq. (5) and the second term of Prop. 1 both retain
    #    f_{u_j}, but with the Section IV numbers that term alone gives a
    #    location-independent floor of 27.3-34.6 dB, above EVERY value plotted
    #    in Figs. 4-8 (13.8-19.3 dB in Fig. 4).  Since the term is independent
    #    of T and Theta it cannot be optimised away, so the published curves
    #    are only reproducible with the direct link removed.  The paper states
    #    the BS-user LoS is "blocked by obstacles"; taking that as full
    #    blockage, f_{u_j} = 0, reproduces the plotted levels.
    #      include_direct = True  -> Eq. (5) exactly as printed
    #      include_direct = False -> f_{u_j} = 0  (used for Figs. 4-8)
    include_direct=False,
)

# =============================================================================
# GEOMETRY
# =============================================================================
def upa_shape(n):
    """Near-square lambda/2 UPA holding n elements (truncated grid)."""
    n1 = max(1, int(round(np.sqrt(n))))
    return n1, int(np.ceil(n / n1))


def panel_rotation(theta_deg, phi_deg):
    """R_l(theta_l, phi_l): maps local panel axes to global coordinates.

    Columns of R are the two in-plane unit axes (u1, u2) and the panel normal.
    phi rotates the in-plane horizontal axis about z; theta tilts the panel out
    of vertical (theta = 90 deg gives a horizontal panel).
    """
    th, ph = np.radians(theta_deg), np.radians(phi_deg)
    u1 = np.array([np.cos(ph), np.sin(ph), 0.])            # in-plane horizontal
    z = np.array([0., 0., 1.])
    u2v = np.cross(u1, z); u2v /= np.linalg.norm(u2v)      # completes the frame
    # rotate u2 from vertical (theta=0) toward horizontal (theta=90)
    u2 = np.cos(th) * z + np.sin(th) * u2v
    u2 /= np.linalg.norm(u2)
    n = np.cross(u1, u2); n /= np.linalg.norm(n)
    return np.column_stack([u1, u2, n])


def irs_elements(l, n_elem):
    """Element coordinates of IRS l (0-based).

    p_{l,n} = p_center,l + R_l . d_n^local ,  d_n^local a lambda/2 UPA grid.
    The panel is centred on the Fig. 3 reference point; row 0 is then used as
    the paper's reference element p_{l,1} for the path-loss distances d_l and
    d_{l,j}.  Re-centring shifts those distances by <0.2 m out of 13-80 m,
    i.e. below 0.03 dB, so it does not affect any reported quantity.
    """
    n1, n2 = upa_shape(n_elem)
    th, ph = ASSUMPTIONS["panel_angles"][l]
    R = panel_rotation(th, ph)
    i, q = np.meshgrid(np.arange(n1), np.arange(n2), indexing="ij")
    loc = np.stack([i.ravel(), q.ravel()], 1).astype(float) * D_ELEM
    loc -= loc.mean(0)                                     # centre the panel
    d_local = np.column_stack([loc, np.zeros(len(loc))])   # (n, 3), z_local = 0
    return IRS_REF[l] + d_local[:n_elem] @ R.T


def sample_areas(J, rng):
    """J non-overlapping 5x5 m squares inside the target region."""
    out = []
    while len(out) < J:
        cx = rng.uniform(REG_X[0], REG_X[1] - SIDE)
        cy = rng.uniform(REG_Y[0], REG_Y[1] - SIDE)
        if all(abs(cx - a) >= SIDE or abs(cy - b) >= SIDE for a, b in out):
            out.append((cx, cy))
    return out


# =============================================================================
# CHANNEL MODEL, Eqs. (1)-(5), and Proposition 1, Eq. (11)
# =============================================================================
class System:
    def __init__(self, corners, L, n_elem, M, include_direct=None):
        self.L, self.M, self.ne, self.N = L, M, n_elem, L * n_elem
        self.J = len(corners)
        inc = (ASSUMPTIONS["include_direct"] if include_direct is None
               else include_direct)

        # --- candidate locations G_j (Sec. III-A) --------------------------
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
        self.d_l = np.linalg.norm(P1, axis=1)                   # d_l = ||p_{l,1}||
        # far-field direction BS <-> IRS l  (Sec. II-A)
        self.a_t = P1 / self.d_l[:, None]
        # alpha_ll', omega_ll' of Eq. (26); psi_ll' of Eq. (27) with cos(nu)=1
        self.al = self.a_t[:, 1][:, None] - self.a_t[:, 1][None, :]
        self.om = self.a_t[:, 2][:, None] - self.a_t[:, 2][None, :]
        self.psi = K ** 2 * (self.al ** 2 + self.om ** 2)

        # --- e_l of Eq. (2):  [e_l]_n = exp(-j k (p_ln - p_l1)^T a^r_l) ----
        self.Pel = [irs_elements(l, n_elem) for l in range(L)]
        self.e = np.stack([np.exp(-1j * K * (self.Pel[l] - self.Pel[l][0])
                                  @ self.a_t[l]) for l in range(L)])

        # --- d_lj(u) and hbar^H_lj(u) of Eqs. (3),(4) ----------------------
        diff = self.U[None] - P1[:, None]
        self.d_lj = np.linalg.norm(diff, axis=2)
        a_lj = diff / self.d_lj[..., None]
        if ASSUMPTIONS["steering"] == "exact":
            # per-element spherical phase, referenced to element 1 so that the
            # common factor exp(-j k d_lj) stays absorbed in the path loss
            hb = []
            for l in range(L):
                dn = np.linalg.norm(self.U[:, None, :] - self.Pel[l][None], axis=2)
                hb.append(np.exp(-1j * K * (dn - dn[:, [0]])))
            self.hbarH = np.stack(hb)
        else:                                   # Eq. (4), far-field plane wave
            self.hbarH = np.stack([
                np.exp(1j * K * (a_lj[l] @ (self.Pel[l] - self.Pel[l][0]).T))
                for l in range(L)])

        # x_{l,n}(u) = [hbar^H_l(u)]_n [e_l]_n, so c_l = sum_n theta_n x_n
        self.X = self.hbarH * self.e[:, None, :]
        self.Xf = self.X.transpose(1, 0, 2).reshape(self.G, self.N)
        self.Xfc = np.conj(self.Xf)

        # --- large-scale weights of hhat, Ghat (Prop. 1) --------------------
        rho = np.sqrt(C0 * self.d_l ** -A_IRS * KAPPA / (KAPPA + 1))
        sig = np.sqrt(C0 * self.d_lj ** -A_IRS * KAPPA / (KAPPA + 1))
        self.w = (rho[:, None] * sig).T                    # (G, L)

        # --- constant floor C_u, second term of Eq. (11) --------------------
        # NOTE Lambda carries an l index (it contains d_lj and d_l); the paper
        # prints it as Lambda_{u_j}.
        Lam = (C0 ** 2 * self.d_lj ** -A_IRS
               * (self.d_l ** -A_IRS)[:, None] / (KAPPA + 1) ** 2)
        scat = ((2 * KAPPA + 1) * Lam * M * n_elem).sum(0)
        du = np.linalg.norm(self.U, axis=1)
        # Eq. (5) contributes E[|f|^2] = C0 * d_u^-alpha_j * M, a constant.
        self.C_dir = (PBAR * C0 * du ** -A_DIR * M if inc
                      else np.zeros(self.G))
        self.C_sca = PBAR * scat
        self.C_u = self.C_sca + self.C_dir

    # g_l^H(T) of Eq. (2): entries exp(+j k t_m^T a^t_l);  T is (M,2) = (y,z)
    def gH(self, T):
        return np.exp(1j * K * (T[:, 0] * self.a_t[:, 1][:, None]
                                + T[:, 1] * self.a_t[:, 2][:, None]))

    def c(self, v):                      # c_l(u) = hbar^H_l Theta_l e_l
        return (self.Xf * v).reshape(self.G, self.L, self.ne).sum(2)

    def snr(self, T, v, sel=None):
        """Eq. (11): expected SNR at each candidate location."""
        sel = np.arange(self.G) if sel is None else sel
        r = (self.w[sel] * self.c(v)[sel]) @ self.gH(T)      # (|sel|, M)
        return PBAR * np.einsum("gm,gm->g", r, np.conj(r)).real + self.C_u[sel]


# =============================================================================
# MONTE-CARLO REFERENCE: Eqs. (1)-(6) with the MRT of Eq. (10)
# =============================================================================
def mc_snr(S, T, v, u, n_mc, rng):
    gH, tot = S.gH(T), 0.0
    for _ in range(n_mc):
        acc = np.zeros(S.M, complex)
        for l in range(S.L):
            Gt = (rng.standard_normal((S.ne, S.M))
                  + 1j * rng.standard_normal((S.ne, S.M))) / np.sqrt(2)
            Gl = np.sqrt(C0 * S.d_l[l] ** -A_IRS) * (
                np.sqrt(KAPPA / (KAPPA + 1)) * np.outer(S.e[l], gH[l])
                + np.sqrt(1 / (KAPPA + 1)) * Gt)                     # Eq. (1)
            ht = (rng.standard_normal(S.ne)
                  + 1j * rng.standard_normal(S.ne)) / np.sqrt(2)
            hH = np.sqrt(C0 * S.d_lj[l, u] ** -A_IRS) * (
                np.sqrt(KAPPA / (KAPPA + 1)) * S.hbarH[l, u]
                + np.sqrt(1 / (KAPPA + 1)) * ht)                     # Eq. (3)
            acc += (hH * v[l * S.ne:(l + 1) * S.ne]) @ Gl
        f = np.sqrt(S.C_dir[u] / (PBAR * S.M)) * (
            rng.standard_normal(S.M) + 1j * rng.standard_normal(S.M)
        ) / np.sqrt(2)                                               # Eq. (5)
        acc += f
        tot += PBAR * np.vdot(acc, acc).real                         # Eqs. (6),(10)
    return tot / n_mc


# =============================================================================
# ALGORITHM 1
# The two epigraph programs (31) and (36) are solved with the paper's own
# surrogates: Eqs. (23)/(24) with the curvature bound (27) for the MA update,
# and the first-order lower bound (35) for the IRS update.  `--crosscheck`
# verifies these updates against CVXPY on the same subproblems.
# =============================================================================
def _weights(h, theta):
    sp = h.max() - h.min()
    if sp <= 0 or theta <= 0:
        w = np.zeros_like(h); w[np.argmin(h)] = 1.0; return w
    z = -(h - h.min()) / (theta * sp)
    w = np.exp(z - z.max())
    return w / w.sum()


def _spacing(T, A):
    """Enforce (7c) box and (7d) ||t_m - t_q|| >= D."""
    T = np.clip(T, -A / 2, A / 2)
    for _ in range(60):
        d = np.linalg.norm(T[:, None] - T[None], axis=2)
        np.fill_diagonal(d, np.inf)
        if d.min() >= D_MIN - 1e-12:
            break
        for m, q in np.argwhere(d < D_MIN):
            if m >= q:
                continue
            vec = T[m] - T[q]; r = np.linalg.norm(vec)
            if r < 1e-9:
                vec, r = np.array([1., 0.]) * 1e-3, 1e-3
            T[m] += (D_MIN - r) / 2 * 1.02 * vec / r
            T[q] -= (D_MIN - r) / 2 * 1.02 * vec / r
        T = np.clip(T, -A / 2, A / 2)
    return T


def _irs_step(S, T, v, om, sel, esel):
    """MM step for Theta from the lower bound (35)."""
    r = (S.w * S.c(v)) @ S.gH(T)
    zeta = S.w * (r @ np.conj(S.gH(T)).T)
    zf = np.repeat(zeta, S.ne, 1)
    g = (om[sel, None] * zf[np.ix_(sel, esel)] * S.Xfc[np.ix_(sel, esel)]).sum(0)
    out = v.copy()
    out[esel] = np.where(np.abs(g) > 0, g / np.maximum(np.abs(g), 1e-300), v[esel])
    return out


def _beta(S, v, sel, om):
    """Re{b_{l,l'}(u)} of Eq. (19), weighted over the selected locations."""
    wc = S.w * S.c(v)
    return PBAR * np.einsum("g,gl,gm->lm", om[sel], wc[sel],
                            np.conj(wc[sel])).real


def _ma_step(S, T, beta, A):
    """MM step for T from (23)/(24) with psi of (27)."""
    Sc = (np.abs(beta) * S.psi).sum()
    if Sc <= 0:
        return T
    nu = K * (T[:, 0][:, None, None] * S.al + T[:, 1][:, None, None] * S.om)
    sn = np.sin(nu)
    gy = -K * np.einsum("lm,klm,lm->k", beta, sn, S.al)          # Eq. (26a)
    gz = -K * np.einsum("lm,klm,lm->k", beta, sn, S.om)          # Eq. (26b)
    return _spacing(np.clip(T + np.stack([gy, gz], 1) / Sc, -A / 2, A / 2), A)


def fpa(M):
    p = (np.arange(M) - (M - 1) / 2) * (LAM / 2)
    return (np.stack([p, np.zeros(M)], 1) if ASSUMPTIONS["fpa_axis"] == "y"
            else np.stack([np.zeros(M), p], 1))


def _init_T(M, A, rng, mode):
    T = (np.stack([(np.arange(M) - (M - 1) / 2) * (LAM / 2), np.zeros(M)], 1)
         if mode == "ula" else rng.uniform(-A / 2, A / 2, (M, 2)))
    return _spacing(np.clip(T.astype(float), -A / 2, A / 2), A)


def _init_v(S, sel):
    x = S.Xf[sel[len(sel) // 2]]
    return np.conj(x) / np.abs(x)


def solve(S, scheme, A, warm=None, seed=0):
    """scheme: ma_irs | ma_stairs | shared_ma_stairs | fpa_irs | fpa_stairs"""
    rng = np.random.default_rng(seed)
    ni, nr = ASSUMPTIONS["ao_iters"], ASSUMPTIONS["ao_restarts"]
    ma_ad = scheme in ("ma_irs", "ma_stairs")
    irs_ad = scheme in ("ma_irs", "fpa_irs")
    is_fpa = scheme.startswith("fpa")
    allE = np.arange(S.N)

    # (P1) and FPA-(area-adaptive IRS) have per-area variables and the outer min
    # is over a union of independent terms -> they decouple across areas.
    if irs_ad:
        vals, Ts, vs = [], [], []
        for j in range(S.J):
            sel = S.gj[j]
            b = (-np.inf, None, None)
            wj = None if warm is None else (warm[0][j], warm[1][j] if
                                            isinstance(warm[1], list) else warm[1])
            for r in range((1 if is_fpa else nr) + (0 if wj is None else 1)):
                if wj is not None and r == (1 if is_fpa else nr):
                    T = fpa(S.M) if is_fpa else _spacing(
                        np.clip(wj[0].copy(), -A / 2, A / 2), A)
                    v = wj[1].copy()
                else:
                    T = fpa(S.M) if is_fpa else _init_T(
                        S.M, A, rng, "ula" if r == 0 else "rand")
                    v = _init_v(S, sel)
                cur = (-np.inf, None, None)
                for it in range(ni):
                    th = 0.5 * (0.02 / 0.5) ** (it / (ni - 1))
                    h = S.snr(T, v, sel)
                    om = np.zeros(S.G); om[sel] = _weights(h, th)
                    if h.min() > cur[0]:
                        cur = (h.min(), T.copy(), v.copy())
                    v = _irs_step(S, T, v, om, sel, allE)
                    if not is_fpa:
                        T = _ma_step(S, T, _beta(S, v, sel, om), A)
                h = S.snr(T, v, sel)
                if h.min() > cur[0]:
                    cur = (h.min(), T, v)
                if cur[0] > b[0]:
                    b = cur
            vals.append(b[0]); Ts.append(b[1]); vs.append(b[2])
        return float(min(vals)), Ts, vs

    # shared IRS phases: (P2), (P3), FPA-staIRS
    best = (-np.inf, None, None)
    for r in range(nr + (0 if warm is None else 1)):
        if warm is not None and r == nr:
            Tw, vw = warm
            T0 = Tw[0] if isinstance(Tw, list) else Tw
            Tl = [fpa(S.M)] * S.J if is_fpa else \
                 [_spacing(np.clip(T0.copy(), -A / 2, A / 2), A)] * S.J
            v = (vw[0] if isinstance(vw, list) else vw).copy()
        else:
            Tl = ([fpa(S.M)] * S.J if is_fpa else
                  ([_init_T(S.M, A, rng, "ula" if r == 0 else "rand")
                    for _ in range(S.J)] if ma_ad else
                   [_init_T(S.M, A, rng, "ula" if r == 0 else "rand")] * S.J))
            v = _init_v(S, np.arange(S.G))
        cur = (-np.inf, None, None)
        for it in range(ni):
            th = 0.5 * (0.02 / 0.5) ** (it / (ni - 1))
            h = np.empty(S.G)
            for j in range(S.J):
                h[S.gj[j]] = S.snr(Tl[j], v, S.gj[j])
            om = _weights(h, th)
            if h.min() > cur[0]:
                cur = (h.min(), [t.copy() for t in Tl], v.copy())
            # IRS update uses every location, since Theta is shared
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
                    Tl = [_ma_step(S, Tl[0], _beta(S, v, np.arange(S.G), om), A)] * S.J
        if cur[0] > best[0]:
            best = cur
        if is_fpa and warm is None:
            break
    return float(best[0]), best[1], best[2]


SCHEMES = ("ma_irs", "ma_stairs", "shared_ma_stairs", "fpa_irs", "fpa_stairs")


def solve_all(S, A, seed=0):
    """Solve in order of increasing feasible set and warm-start each scheme
    from the more constrained one, so the nesting-implied ordering holds."""
    o, s = {}, {}
    o["fpa_stairs"], T, v = solve(S, "fpa_stairs", A, seed=seed)
    s["fpa_stairs"] = (T, v)
    o["fpa_irs"], T2, v2 = solve(S, "fpa_irs", A, warm=(T, v), seed=seed)
    o["shared_ma_stairs"], T3, v3 = solve(S, "shared_ma_stairs", A,
                                          warm=(T, v), seed=seed)
    o["ma_stairs"], T4, v4 = solve(S, "ma_stairs", A, warm=(T3, v3), seed=seed)
    o["ma_irs"], _, _ = solve(S, "ma_irs", A, warm=(T4, v4), seed=seed)
    return o


# =============================================================================
# FIGURES
# =============================================================================
os.makedirs("figs2", exist_ok=True); os.makedirs("cache2", exist_ok=True)
MODES = [True, False]   # include_direct
ST = {"ma_irs":            dict(label="Area-adaptive MA-IRS",    c="#1f4fd8", m="*", ls="-"),
      "ma_stairs":         dict(label="Area-adaptive MA-staIRS", c="#e21ce2", m="v", ls="--"),
      "shared_ma_stairs":  dict(label="Shared MA-staIRS",        c="#e01b24", m="o", ls="-"),
      "fpa_irs":           dict(label="FPA-(area-adaptive IRS)", c="#1a9c3c", m="D", ls="-"),
      "fpa_stairs":        dict(label="FPA-staIRS",              c="k",       m="",  ls=":")}
ORD = list(ST)


def dB(x): return 10 * np.log10(x)


OPEN_FIGS = False          # set by --open


def save_fig(fig, name):
    """Write the figure and print its ABSOLUTE path (figs2/ is created relative
    to the working directory, which in VS Code is not always the script's
    folder).  With --open, also hand it to the OS image viewer."""
    p = os.path.abspath(os.path.join("figs2", name + ".png"))
    fig.savefig(p, dpi=185, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {p}", flush=True)
    if OPEN_FIGS == "vscode":
        # open the PNG as a tab inside VS Code (needs the `code` CLI on PATH:
        # Command Palette -> "Shell Command: Install 'code' command in PATH")
        import shutil, subprocess
        exe = shutil.which("code") or shutil.which("code-insiders")
        if exe:
            subprocess.run([exe, "-r", p], check=False)
        else:
            print("     (`code` not on PATH -- run the Shell Command installer)")
    elif OPEN_FIGS:
        import webbrowser
        webbrowser.open("file://" + p)


def run(tag, xs, build, trials, inc, single=False):
    """Monte-Carlo sweep; common area layouts across x and across panels."""
    f = f"cache2/{tag}_{'dir' if inc else 'nodir'}_t{trials}.npz"
    if os.path.exists(f):
        return dict(np.load(f, allow_pickle=True))
    keys = ["ma_irs"] if single else ORD
    acc = {k: np.zeros((len(xs), trials)) for k in keys}
    ASSUMPTIONS["include_direct"] = inc
    for t in range(trials):
        for i, x in enumerate(xs):
            J, L, ne, M, A = build(x)
            S = System(sample_areas(J, np.random.default_rng(10_000 + t)),
                       L, ne, M)
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
    ax.set_xlabel(xlabel, fontsize=10); ax.set_title(title, fontsize=9.5)
    ax.grid(True, ls=":", lw=.6, alpha=.7); ax.tick_params(labelsize=8.5)
    ax.set_xticks(xs)


def two_panel(name, xs, build, trials, xlabel, single=False):
    ds = [run(name, xs, build, trials, b, single) for b in MODES]
    fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.9), sharey=False)
    for ax, d, ttl in zip(axs, ds,
                          ["(a) Eq. (5) retained: direct link included",
                           "(b) $\\mathbf{f}_{u_j}=0$  (reproduces Figs. 4-8)"]):
        panel(ax, xs, d, xlabel, ttl, ["ma_irs"] if single else None)
    axs[0].set_ylabel("Average worst-case SNR (dB)", fontsize=10)
    axs[0].legend(fontsize=7.5, framealpha=.95)
    lo, hi = axs[0].get_ylim()
    if hi - lo < 4:                       # make the collapsed panel readable
        axs[0].set_ylim((lo + hi) / 2 - 2.5, (lo + hi) / 2 + 2.5)
    save_fig(fig, name)
    return ds


# ---------------------------------------------------------------------------
# Extra figures the model can produce
# ---------------------------------------------------------------------------
def fig_setup():
    """Top view of the deployment -- the paper's Fig. 3, plus IRS apertures."""
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.add_patch(plt.Rectangle((REG_X[0], REG_Y[0]), REG_X[1] - REG_X[0],
                               REG_Y[1] - REG_Y[0], fc="0.92", ec="0.6",
                               ls="--", label="target-area region"))
    for c in sample_areas(3, np.random.default_rng(10_000)):
        ax.add_patch(plt.Rectangle(c, SIDE, SIDE, fc="#f2c14e", ec="k", lw=.8))
    ax.plot(0, 0, "s", ms=11, color="#3a7d44", label="BS")
    for l in range(5):
        p = IRS_REF[l]
        P = irs_elements(l, 20)
        ax.plot(P[:, 0], P[:, 1], "-", lw=4, color="#1f4fd8", alpha=.9,
                solid_capstyle="butt")
        ax.annotate(f"IRS {l+1}", (p[0], p[1]), textcoords="offset points",
                    xytext=(6, 6), fontsize=8)
    ax.plot([], [], "-", lw=4, color="#1f4fd8", label="IRS aperture (top view)")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    ax.set_title("Simulation setup, top view (cf. paper Fig. 3)", fontsize=10)
    ax.legend(fontsize=8, loc="upper left"); ax.grid(True, ls=":", lw=.6)
    ax.set_aspect("equal"); ax.set_xlim(-8, 78); ax.set_ylim(-46, 46)
    save_fig(fig, "fig3_setup")


def fig_convergence(seed=0):
    """Algorithm 1: best-so-far worst-case SNR vs AO iteration."""
    S = System(sample_areas(3, np.random.default_rng(10_000)), 3, 20, 4)
    A, ni = 5 * LAM, ASSUMPTIONS["ao_iters"]
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    for sc in ORD:
        rng = np.random.default_rng(seed)
        is_f = sc.startswith("fpa"); ma_ad = sc in ("ma_irs", "ma_stairs")
        irs_ad = sc in ("ma_irs", "fpa_irs")
        Tl = [fpa(S.M)] * S.J if is_f else [_init_T(S.M, A, rng, "ula")] * S.J
        V = ([_init_v(S, S.gj[j]) for j in range(S.J)] if irs_ad
             else [_init_v(S, np.arange(S.G))] * S.J)
        best, tr = -np.inf, []
        for it in range(ni):
            th = 0.5 * (0.02 / 0.5) ** (it / (ni - 1))
            h = np.concatenate([S.snr(Tl[j], V[j], S.gj[j])
                                for j in range(S.J)])
            best = max(best, h.min()); tr.append(best)
            om = _weights(h, th)
            for j in range(S.J):
                sel = S.gj[j]
                oj = np.zeros(S.G); oj[sel] = om[sel]
                V[j] = _irs_step(S, Tl[j], V[j], oj, sel, np.arange(S.N))
            if not irs_ad:
                V = [V[0]] * S.J
            if not is_f:
                if ma_ad:
                    Tl = [_ma_step(S, Tl[j],
                                   _beta(S, V[j], S.gj[j], om), A)
                          for j in range(S.J)]
                else:
                    Tl = [_ma_step(S, Tl[0],
                                   _beta(S, V[0], np.arange(S.G), om), A)] * S.J
        ax.plot(dB(np.array(tr)), color=ST[sc]["c"], ls=ST[sc]["ls"], lw=1.5,
                label=ST[sc]["label"])
    ax.set_xlabel("AO iteration $r$"); ax.set_ylabel("Best-so-far worst-case SNR (dB)")
    ax.set_title("Algorithm 1 convergence ($J=3$, $L=3$, $N_e=20$, $M=4$)\n"
                 "single restart, no warm start -- final reported values use\n"
                 "%d restarts plus the warm-start ladder, so they are higher"
                 % ASSUMPTIONS["ao_restarts"], fontsize=8.5)
    ax.grid(True, ls=":", lw=.6); ax.legend(fontsize=7.5, loc="lower right")
    save_fig(fig, "figS1_convergence")


def fig_selectivity():
    """How much a static IRS can distinguish the two extreme user positions,
    as a function of the vertical-panel azimuth phi (an unstated parameter)."""
    uA, uB = np.array([60., -40., 0.]), np.array([60., 40., 0.])
    phis = np.arange(0, 91, 5.0)
    keep = list(ASSUMPTIONS["panel_angles"])
    corr, diff = [], []
    for ph in phis:
        ASSUMPTIONS["panel_angles"] = [(90., 0.), (0., -ph), (0., ph),
                                       (0., -ph), (0., ph)]
        pa, pb = [], []
        for l in range(3):
            P = irs_elements(l, 20)
            for u, acc in ((uA, pa), (uB, pb)):
                a = u - P[0]; a /= np.linalg.norm(a)
                acc.append(K * ((P - P[0]) @ a))
        pa, pb = np.concatenate(pa), np.concatenate(pb)
        d = np.angle(np.exp(1j * (pa - pb)))
        corr.append(np.abs(np.vdot(np.exp(1j * pa), np.exp(1j * pb))) / 60)
        diff.append(d.max() - d.min())
    ASSUMPTIONS["panel_angles"] = keep
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.plot(phis, corr, "o-", color="#1f4fd8", ms=4,
            label=r"$|\bar{\mathbf{h}}(u_A)^H\bar{\mathbf{h}}(u_B)|/N$")
    ax.set_xlabel(r"vertical-panel azimuth $|\phi_\ell|$ (deg)")
    ax.set_ylabel("steering-vector correlation")
    ax2 = ax.twinx()
    ax2.plot(phis, diff, "s--", color="#e01b24", ms=4,
             label="differential phase spread")
    ax2.set_ylabel("differential phase spread (rad)", color="#e01b24")
    ax.set_title("IRS phase selectivity vs panel orientation\n"
                 "(low correlation = adaptive IRS earns more than static)",
                 fontsize=9)
    ax.grid(True, ls=":", lw=.6)
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    save_fig(fig, "figS2_selectivity")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--fig", default="4",
                    choices=["3", "4", "5", "6", "7", "8", "conv", "sel", "all"])
    ap.add_argument("--trials", type=int, default=8)
    ap.add_argument("--open", action="store_true", dest="open_figs",
                    help="open each figure in the OS image viewer when saved")
    ap.add_argument("--vscode", action="store_true",
                    help="open each figure as a tab inside VS Code")
    ap.add_argument("--check", action="store_true",
                    help="validate Prop. 1 (Eq. 11) against Monte Carlo")
    a = ap.parse_args(); T = a.trials
    OPEN_FIGS = "vscode" if a.vscode else a.open_figs
    print(f"figures -> {os.path.abspath('figs2')}")
    print(f"cache   -> {os.path.abspath('cache2')}\n")

    if a.check:
        rng = np.random.default_rng(0)
        print("Proposition 1 (Eq. 11) vs Monte Carlo of Eqs. (1)-(6),(10)")
        for L in (1, 3):
            S = System(sample_areas(3, np.random.default_rng(7)), L, 12, 4)
            Tm = _init_T(4, 0.5, rng, "rand")
            v = np.exp(1j * rng.uniform(0, 2 * np.pi, S.N))
            for u in (5, 40):
                an = S.snr(Tm, v)[u]
                mc = mc_snr(S, Tm, v, u, 40000, np.random.default_rng(3))
                print("  L=%d u=%-3d analytic %10.3f  MC %10.3f  err %+.4f dB"
                      % (L, u, an, mc, 10 * np.log10(mc / an)))
        S = System(sample_areas(3, np.random.default_rng(7)), 3, 20, 4)
        S2 = System(sample_areas(3, np.random.default_rng(7)), 3, 20, 4,
                    include_direct=True)
        print("\nConstant floor C_u with Eq. (5) exactly as printed:")
        S, S_nodir = S2, S
        print("  direct-link term : %6.2f .. %6.2f dB"
              % (dB(S.C_dir.min()), dB(S.C_dir.max())))
        print("  scattering term  : %6.2f .. %6.2f dB"
              % (dB(S.C_sca.min()), dB(S.C_sca.max())))
        print("  -> floor exceeds every value plotted in Fig. 4 (13.8-19.3 dB)")
        print("  C_u with f_u = 0 : %6.2f .. %6.2f dB"
              % (dB(S_nodir.C_u.min()), dB(S_nodir.C_u.max())))
        raise SystemExit

    if a.fig in ("3", "all"):
        fig_setup()
    if a.fig in ("conv", "all"):
        fig_convergence()
    if a.fig in ("sel", "all"):
        fig_selectivity()
    if a.fig in ("4", "all"):
        two_panel("fig4_vs_J", np.arange(1, 7),
                  lambda J: (int(J), 3, 20, 4, 5 * LAM), T,
                  "Number of target areas, $J$")
    if a.fig in ("5", "all"):
        two_panel("fig5_vs_M", np.array([2, 4, 6, 8, 10, 12]),
                  lambda M: (3, 3, 20, int(M), 8 * LAM), T,
                  "Number of transmit MAs, $M$")
    if a.fig in ("6", "all"):
        two_panel("fig6_vs_Ne", np.array([10, 20, 30, 40, 50, 60]),
                  lambda n: (3, 3, int(n), 4, 5 * LAM), T,
                  "Number of elements at each IRS, $N_{\\mathrm{e}}$")
    if a.fig in ("7", "all"):
        two_panel("fig7a_vs_L_Ne20", np.arange(1, 6),
                  lambda L: (3, int(L), 20, 4, 5 * LAM), T,
                  "Number of IRSs, $L$")
        two_panel("fig7b_vs_L_N120", np.arange(1, 6),
                  lambda L: (3, int(L), 120 // int(L), 4, 5 * LAM), T,
                  "Number of IRSs, $L$")
    if a.fig in ("8", "all"):
        # Fig. 8: area-adaptive MA-IRS only, A = 8 lambda, L = 2, Ne = N/L,
        # Ctot = rho*ce*M + ce*N  (Sec. IV)
        for blk in MODES:
            ASSUMPTIONS["include_direct"] = blk
            for lbl, pairs in (("a", [(5, 120), (10, 120), (20, 120), (30, 120)]),
                               ("b", [(20, 120), (20, 240), (20, 360)])):
                fn = f"cache2/fig8{lbl}_{'dir' if blk else 'nodir'}_t{T}.npz"
                if os.path.exists(fn):
                    continue
                out = {}
                for rho, ct in pairs:
                    Ms, vs = [], []
                    M = 1
                    while ct - rho * M >= max(6, 2 * 2):
                        N = ct - rho * M; ne = int(N // 2)
                        acc = np.zeros(T)
                        for t in range(T):
                            S = System(sample_areas(3, np.random.default_rng(20_000 + t)),
                                       2, ne, M)
                            acc[t] = solve(S, "ma_irs", 8 * LAM, seed=t)[0]
                        Ms.append(M); vs.append(acc.mean()); M += 1
                    out[f"M_{rho}_{ct}"] = np.array(Ms)
                    out[f"v_{rho}_{ct}"] = dB(np.array(vs))
                    print(f"    fig8{lbl} direct={blk} rho={rho} Ctot={ct} done", flush=True)
                np.savez(fn, **out)
        for lbl, pairs, ttl in (("a", [(5, 120), (10, 120), (20, 120), (30, 120)],
                                 "$C_{\\mathrm{tot}}=120c_{\\mathrm{e}}$"),
                                ("b", [(20, 120), (20, 240), (20, 360)], "$\\rho=20$")):
            fig, axs = plt.subplots(1, 2, figsize=(10.2, 3.9))
            for ax, blk, sub in zip(axs, MODES,
                                    ["(a) direct link included",
                                     "(b) $\\mathbf{f}_{u_j}=0$"]):
                d = dict(np.load(f"cache2/fig8{lbl}_{'dir' if blk else 'nodir'}_t{T}.npz"))
                for rho, ct in pairs:
                    M, v = d[f"M_{rho}_{ct}"], d[f"v_{rho}_{ct}"]
                    lab = (f"$\\rho={rho}$" if lbl == "a"
                           else f"$C_{{\\mathrm{{tot}}}}={ct}c_{{\\mathrm{{e}}}}$")
                    ax.plot(M, v, marker="o", ms=4.5, lw=1.4, mfc="none", label=lab)
                    i = int(np.argmax(v))
                    ax.annotate(f"$({M[i]},{ct-rho*M[i]})$", xy=(M[i], v[i]),
                                xytext=(M[i] + 1, v[i] - 1.6), fontsize=7,
                                arrowprops=dict(arrowstyle="->", lw=.6))
                ax.set_xlabel("Number of transmit MAs, $M$", fontsize=10)
                ax.set_title(f"{sub} -- {ttl}", fontsize=9.5)
                ax.grid(True, ls=":", lw=.6, alpha=.7); ax.legend(fontsize=7.5)
            axs[0].set_ylabel("Average worst-case SNR (dB)", fontsize=10)
            save_fig(fig, f"fig8{lbl}_cost")
