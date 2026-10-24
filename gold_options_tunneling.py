# =====================================================================================
#  GOLD OPTIONS · TUNNELING ENGINE
#  1D time-dependent Schrodinger solver (split-step Fourier) + exact Wigner transform.
#  Price wave-packet  psi(x,t),  x = ln(K/S_ref)/sigma_ref   (position in daily-sigma units)
#  Dealer-gamma strikes (call wall / put wall) = potential barriers V(x).
#  Daily filter:  evolve (prediction)  ->  Gaussian measurement at the observed close (update)
#  Panels:  [1] outside 3D phase-space landscape W(x,p)   [2] Wigner heatmap   [3] signal heatmap
#  Output:  32 s MP4.   pip install numpy scipy pandas matplotlib yfinance imageio imageio-ffmpeg
# =====================================================================================
import os, time, warnings
import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import LinearSegmentedColormap, LightSource
from scipy.ndimage import gaussian_filter
from scipy.stats import norm
warnings.filterwarnings("ignore")

CFG = dict(
    N_DAYS=52,            # trading days replayed (heatmap reveals one day at a time)
    FPS=24, SECONDS=32.0, # total video length
    LIVE_SECONDS=3.5,     # last seconds = "today" with live GLD option walls (if reachable)
    W=1080, H=1920, DPI=100,
    HBAR=0.45,            # effective Planck constant  (market uncertainty / path roughness)
    SIGMA0=0.70,          # initial packet width (sigma units)
    MEAS=1.00,            # measurement (close print) width
    STRIKE_STEP=50,       # strike grid used for the historical barrier proxy
    V_BASE=0.90, WALL_W=0.50,
    LIVE_GEX=True,        # try to pull GLD option chain (free, Yahoo) for today's walls
    OUT="gold_options_tunneling.mp4",
    PREVIEW_FRAMES=None,  # e.g. 48 to test quickly
)

BG = "#04060a"; INK = "#cfe9ee"; DIM = "#5f7c85"; AMBER = "#ffb23e"; CYAN = "#41e6d8"

# ------------------------------------------------------------------ colour system
WIG = LinearSegmentedColormap.from_list("wig", [
    (0.00, "#8a63ff"), (0.30, "#2b1c8c"), (0.45, "#0a0830"), (0.50, "#03060a"),
    (0.56, "#06303a"), (0.72, "#13a9a6"), (0.88, "#7ff2e4"), (1.00, "#ffffff")])
LUT = (WIG(np.linspace(0, 1, 2048))[:, :3]).astype(np.float32)
PHASE = LinearSegmentedColormap.from_list("phase", [
    (0.00, "#32145d"), (0.30, "#120d25"), (0.48, "#09090b"),
    (0.55, "#53183b"), (0.76, "#f06a22"), (1.00, "#fff2a1")])
MARKET_HEAT = LinearSegmentedColormap.from_list("market_heat", [
    (0.00, "#3544cc"), (0.25, "#287fd5"), (0.50, "#28d8ca"),
    (0.72, "#43ed76"), (0.88, "#f1e83c"), (1.00, "#f34b43")])

def to_rgb(Wv, scale, gamma=0.55):
    v = np.sign(Wv) * (np.abs(Wv) / scale) ** gamma
    v = np.clip(v, -1, 1)
    return LUT[((v + 1) * 0.5 * 2047).astype(np.int32)]

# ------------------------------------------------------------------ data
def synthetic_gold(n=520, seed=11):
    rng = np.random.default_rng(seed)
    vol, px, rows = 0.009, 3200.0, []
    for _ in range(n):
        z = rng.standard_t(5) / 1.29
        vol = 0.92 * vol + 0.08 * 0.009 + 0.0012 * abs(z)
        r = 0.0005 + vol * z
        o = px * (1 + 0.15 * vol * rng.standard_normal()); c = px * np.exp(r)
        h = max(o, c) * (1 + abs(rng.standard_normal()) * 0.4 * vol)
        l = min(o, c) * (1 - abs(rng.standard_normal()) * 0.4 * vol)
        rows.append((o, h, l, c)); px = c
    idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n)
    return pd.DataFrame(rows, columns=["Open", "High", "Low", "Close"], index=idx)

def load_gold():
    try:
        import yfinance as yf
        df = yf.download("GC=F", period="2y", interval="1d", auto_adjust=False, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df[["Open", "High", "Low", "Close"]].dropna()
        assert len(df) > 200
        return df, "GC=F · Yahoo (free)"
    except Exception as e:
        print("[data] Yahoo unavailable -> synthetic gold series for the demo:", repr(e)[:80])
        return synthetic_gold(), "SYNTHETIC DEMO SERIES"

def live_gld_walls(gc_spot):
    """Call/Put walls from the live GLD chain: dealer GEX per strike (BS gamma * OI), mapped to GC."""
    try:
        import yfinance as yf
        t = yf.Ticker("GLD")
        spot = float(t.history(period="5d")["Close"].dropna().iloc[-1])
        rows = []
        for e in t.options[:5]:
            T = max((pd.Timestamp(e) - pd.Timestamp.today()).days, 1) / 365.0
            ch = t.option_chain(e)
            for df, sgn in ((ch.calls, 1.0), (ch.puts, -1.0)):
                K = df.strike.values.astype(float); oi = df.openInterest.fillna(0).values.astype(float)
                iv = df.impliedVolatility.values.astype(float)
                ok = (iv > 0.02) & (oi > 0)
                K, oi, iv = K[ok], oi[ok], iv[ok]
                d1 = (np.log(spot / K) + 0.5 * iv ** 2 * T) / (iv * np.sqrt(T))
                gam = norm.pdf(d1) / (spot * iv * np.sqrt(T))
                rows.append(pd.DataFrame({"K": K, "g": sgn * gam * oi * 100 * spot ** 2 * 0.01}))
        g = pd.concat(rows).groupby("K").g.sum()
        up, dn = g[g.index > spot * 1.003], g[g.index < spot * 0.997]
        kc, kp = up.idxmax(), dn.idxmin()
        mx = float(g.abs().max())
        ratio = gc_spot / spot
        return dict(Kc=kc * ratio, Kp=kp * ratio,
                    Vc=float(np.clip(0.45 + 1.6 * abs(up[kc]) / mx, 0.4, 2.2)),
                    Vp=float(np.clip(0.45 + 1.6 * abs(dn[kp]) / mx, 0.4, 2.2)))
    except Exception as e:
        print("[live] GLD chain unavailable -> proxy walls only:", repr(e)[:80])
        return None

# ------------------------------------------------------------------ physics
class Schrodinger1D:
    def __init__(self, xmin, xmax, hbar, dx=0.03):
        self.N = 1 << int(np.ceil(np.log2((xmax - xmin) / dx)))
        self.dx, self.hbar = dx, hbar
        self.x = xmin + dx * np.arange(self.N)
        self.k = 2 * np.pi * np.fft.fftfreq(self.N, d=dx)
        e = 5.0
        ramp = np.maximum(0, e - (self.x - self.x[0])) + np.maximum(0, e - (self.x[-1] - self.x))
        self.mask = np.exp(-0.02 * (ramp / e) ** 2)
        self.psi = None

    def init_packet(self, x0, s0, p0):
        self.psi = ((2 * np.pi * s0 ** 2) ** -0.25 * np.exp(-(self.x - x0) ** 2 / (4 * s0 ** 2)
                    + 1j * p0 * (self.x - x0) / self.hbar)).astype(np.complex128)

    def step(self, V, dt):
        hb = self.hbar
        h = np.exp(-0.5j * V * dt / hb)
        self.psi *= h
        self.psi = np.fft.ifft(np.fft.fft(self.psi) * np.exp(-0.5j * hb * self.k ** 2 * dt))
        self.psi *= h
        self.psi *= self.mask

    def normalise(self):
        self.psi /= np.sqrt(np.sum(np.abs(self.psi) ** 2) * self.dx)

    def measure(self, x_obs, s):
        self.psi *= np.exp(-(self.x - x_obs) ** 2 / (4 * s ** 2)); self.normalise()

    def mean_p(self):
        w = np.abs(np.fft.fft(self.psi)) ** 2
        return self.hbar * float(np.sum(self.k * w) / np.sum(w))

    def kick(self, dp, xref):
        self.psi *= np.exp(1j * dp * (self.x - xref) / self.hbar)

    def dens(self): return np.abs(self.psi) ** 2
    def mean_x(self):
        d = self.dens(); return float(np.sum(self.x * d) / np.sum(d))

def barrier(x, xc, w, V0, eps=0.05):
    return V0 * 0.5 * (np.tanh((x - (xc - w / 2)) / eps) - np.tanh((x - (xc + w / 2)) / eps))

def wigner(eng, xc, Nw=192, stride=3, K=96, pmax=3.2):
    dxw = eng.dx * stride
    c = int(round((xc - eng.x[0]) / eng.dx))
    idx = np.clip(c + stride * (np.arange(Nw) - Nw // 2), 0, eng.N - 1)
    psi = eng.psi[idx]
    pad = 2 * K
    pp = np.concatenate([np.zeros(pad), psi, np.zeros(pad)])
    kk = np.fft.fftfreq(2 * K, 1.0 / (2 * K)).astype(int)
    i = np.arange(Nw)[:, None]
    C = np.conj(pp[pad + i + kk[None, :]]) * pp[pad + i - kk[None, :]]
    Wf = np.fft.fftshift((np.fft.ifft(C, axis=1) * (2 * K)).real, axes=1) * dxw / (np.pi * eng.hbar)
    p = np.pi * eng.hbar * np.arange(-K, K) / (2 * K * dxw)
    sel = (p >= -pmax) & (p <= pmax)
    xw = eng.x[0] + (c - stride * (Nw // 2)) * eng.dx + dxw * np.arange(Nw)
    return xw, p[sel], Wf[:, sel]

# ------------------------------------------------------------------ day model
def build_days(df, cfg):
    r = np.log(df.Close).diff()
    sig = r.ewm(span=20).std()
    rvp = sig.rolling(120, min_periods=40).rank(pct=True).fillna(0.5)
    n = cfg["N_DAYS"]
    d0 = len(df) - n
    sref = float(np.median(sig.iloc[d0:]))
    Sref = float(df.Close.iloc[d0])
    days = []
    for j in range(d0, len(df)):
        S = float(df.Close.iloc[j]); sd = float(sig.iloc[j])
        m5 = float(r.iloc[j - 4:j + 1].mean())
        p = float(np.clip(2.0 * m5 / sref, -2.5, 2.5))
        st = cfg["STRIKE_STEP"]
        Kc = np.ceil(S * (1 + 0.6 * sd) / st) * st
        Kp = np.floor(S * (1 - 0.6 * sd) / st) * st
        f = lambda K: cfg["V_BASE"] * (1 + 0.6 * (K % 100 == 0)) * (1.25 - 0.7 * rvp.iloc[j])
        days.append(dict(date=df.index[j], S=S, sd=sd, a=float(np.clip(sd / sref, 0.6, 1.7)), p=p, Kc=Kc, Kp=Kp, Vc=f(Kc), Vp=f(Kp),
                         o=float(df.Open.iloc[j]), h=float(df.High.iloc[j]), l=float(df.Low.iloc[j]),
                         src="STRIKE-GRID PROXY"))
    return days, Sref, sref

def regime(E, V0, T):
    if E >= V0: return "OVER_BARRIER", "#ffffff"
    if T >= 0.15: return "TUNNELING", CYAN
    if T >= 0.02: return "EVANESCENT", "#6f7dff"
    return "CONFINED", "#52656b"

# ------------------------------------------------------------------ writer
class Writer:
    def __init__(self, path, fps, w, h):
        self.k = None
        try:
            import imageio.v2 as iio
            self.wr = iio.get_writer(path, fps=fps, codec="libx264", pixelformat="yuv420p",
                                     macro_block_size=1, ffmpeg_params=["-crf", "16", "-preset", "medium"])
            self.k = "imageio"
        except Exception as e:
            import cv2
            self.wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h)); self.k = "cv2"
    def add(self, f):
        if self.k == "imageio": self.wr.append_data(f)
        else: self.wr.write(f[..., ::-1])
    def close(self):
        self.wr.close() if self.k == "imageio" else self.wr.release()

# ------------------------------------------------------------------ render
def style3d(ax):
    ax.set_facecolor(BG)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.pane.set_facecolor((0, 0, 0, 0)); a.pane.set_edgecolor((1, 1, 1, 0.06))
        a._axinfo["grid"]["color"] = (0.45, 0.8, 0.85, 0.10); a._axinfo["grid"]["linewidth"] = 0.5
        a.set_tick_params(colors=DIM, labelsize=7, pad=-3)
    ax.set_xlabel("x · position  ln(K/S)/σ", color=DIM, fontsize=8, labelpad=-6)
    ax.set_ylabel("p · momentum", color=DIM, fontsize=8, labelpad=-6)
    ax.set_zlabel("W(x,p)", color=DIM, fontsize=8, labelpad=-8)

def main(cfg=CFG):
    t0 = time.time()
    df, src = load_gold()
    days, Sref, sref = build_days(df, cfg)
    live = None
    if cfg["LIVE_GEX"] and src.startswith("GC"):
        live = live_gld_walls(days[-1]["S"])
    xobs = [np.log(d["S"] / Sref) / sref for d in days]
    xK = lambda K: np.log(K / Sref) / sref
    xlo = min(min(xobs), min(xK(d["Kp"]) for d in days)) - 14
    xhi = max(max(xobs), max(xK(d["Kc"]) for d in days)) + 14
    eng = Schrodinger1D(xlo, xhi, cfg["HBAR"])
    FPS, TOT = cfg["FPS"], int(cfg["FPS"] * cfg["SECONDS"])
    LIVEF = int(cfg["FPS"] * cfg["LIVE_SECONDS"]) if live else 0
    FPD = max(4, (TOT - LIVEF) // cfg["N_DAYS"])
    NF = FPD * cfg["N_DAYS"] + LIVEF
    if cfg["PREVIEW_FRAMES"]: NF = min(NF, cfg["PREVIEW_FRAMES"])
    SUB = 6; dt = 1.0 / FPD / SUB
    w_wall = cfg["WALL_W"]

    Wpx, Hpx = cfg["W"], cfg["H"]
    fig = Figure(figsize=(Wpx / cfg["DPI"], Hpx / cfg["DPI"]), dpi=cfg["DPI"], facecolor=BG)
    FigureCanvasAgg(fig)
    ov = fig.add_axes([0, 0, 1, 1]); ov.set_xlim(0, 1); ov.set_ylim(0, 1); ov.axis("off"); ov.set_zorder(10)
    ax3 = fig.add_axes([0.0, 0.505, 1.0, 0.42], projection="3d", facecolor=BG)
    axt = fig.add_axes([0.105, 0.272, 0.76, 0.185], facecolor="#f5f3ee")
    axt.set_facecolor("#f5f3ee")
    axt.tick_params(colors="#312a35", labelsize=8)
    axt.set_xlabel("x · position  ln(K/S)/σ", color="#312a35", fontsize=8)
    axt.set_ylabel("p · momentum", color="#312a35", fontsize=8)
    for spine in axt.spines.values():
        spine.set_color("#bdb7ad")
    cax = fig.add_axes([0.88, 0.292, 0.018, 0.145])
    cax.set_facecolor("#f5f3ee")
    axc = fig.add_axes([0.105, 0.055, 0.77, 0.16], facecolor=BG)
    cax_market = fig.add_axes([0.895, 0.072, 0.018, 0.125])
    market_cmap = MARKET_HEAT.copy()
    market_cmap.set_bad(BG)
    market_img = axc.imshow(np.ma.masked_all((4, cfg["N_DAYS"])), origin="lower", aspect="auto",
                            extent=(-0.5, cfg["N_DAYS"] - 0.5, -0.5, 3.5),
                            cmap=market_cmap, vmin=0, vmax=1, interpolation="nearest")
    market_bar = fig.colorbar(market_img, cax=cax_market, ticks=[0, 0.5, 1])
    market_bar.set_label("signal", color=DIM, fontsize=7)
    market_bar.ax.tick_params(colors=DIM, labelsize=6, length=2)
    axc.set_yticks(range(4), ["Momentum", "Rel. volatility", "Call wall", "Put wall"])
    axc.tick_params(axis="y", colors=INK, labelsize=7, length=0, pad=4)
    axc.tick_params(axis="x", colors=DIM, labelsize=7, length=2)
    for spine in axc.spines.values():
        spine.set_color("#1b2c33")
    heat_cursor = axc.axvline(0, color="#fff4cf", lw=0.8, alpha=0.9)
    axc.set_xlim(-0.5, cfg["N_DAYS"] - 0.5)
    axc.set_ylim(-0.5, 3.5)
    im = axt.imshow(np.zeros((192, 192)), origin="lower", aspect="auto", cmap=PHASE,
                    vmin=-0.35, vmax=1.0, interpolation="bilinear")
    fig.colorbar(im, cax=cax, label="W / scale")
    cax.yaxis.label.set_color("#312a35")
    cax.tick_params(colors="#312a35", labelsize=7)
    phase_trail = []
    phase_line, = axt.plot([], [], color="#45c9bd", lw=1.2, alpha=0.85)
    phase_point = axt.scatter([], [], s=34, c="#fff7dc", edgecolors="#3b97a4",
                              linewidths=1.2, zorder=5)
    call_line = axt.axvline(0, color="#d98425", lw=1, ls=(0, (4, 3)), alpha=0.9)
    put_line = axt.axvline(0, color="#677cc7", lw=1, ls=(0, (4, 3)), alpha=0.9)

    # static chrome
    ov.text(0.04, 0.965, "GOLD OPTIONS · TUNNELING ENGINE", color=INK, fontsize=21, fontweight="bold", va="center")
    sub = ov.text(0.04, 0.943, "", color=DIM, fontsize=9.5, family="monospace", va="center")
    ov.text(0.96, 0.965, "COMEX GC", color=AMBER, fontsize=11, family="monospace", ha="right", va="center")
    ov.plot([0.04, 0.96], [0.931, 0.931], color="#17323a", lw=0.8)
    ov.text(0.04, 0.485, "02 · PHASE SPACE  —  Wigner distribution, moving packet, and option walls",
            color=DIM, fontsize=8.5, family="monospace")
    ov.text(0.04, 0.2255, "03 · MARKET SIGNALS  —  momentum, volatility, and wall proximity", color=DIM, fontsize=8.5, family="monospace")
    ov.text(0.04, 0.926, "01 · OUTSIDE VIEW  —  phase-space landscape", color=DIM, fontsize=8.5, family="monospace", va="top")
    hud1 = axt.text(0.012, 0.965, "", transform=axt.transAxes, color="#19161a", fontsize=9, family="monospace",
                    va="top", fontweight="bold",
                    bbox=dict(facecolor="#f5f3ee", edgecolor="none", alpha=0.78, pad=2))
    hud2 = axt.text(0.012, 0.84, "", transform=axt.transAxes, color="#19161a", fontsize=8, family="monospace",
                    va="top", bbox=dict(facecolor="#f5f3ee", edgecolor="none", alpha=0.78, pad=2))
    hud3 = axt.text(0.988, 0.965, "", transform=axt.transAxes, color="#7a4517", fontsize=8, family="monospace",
                    va="top", ha="right")
    foot = ov.text(0.04, 0.018, "", color="#3f5961", fontsize=7.5, family="monospace")

    wr = Writer(cfg["OUT"], FPS, Wpx, Hpx)
    ls = LightSource(azdeg=300, altdeg=38)
    trail = []
    dirn = 1.0; Vprev = None; Vday = None
    pt = 0.0; di = 0; cur = None
    for f in range(NF):
        in_live = f >= FPD * cfg["N_DAYS"]
        di_new = min(f // FPD, cfg["N_DAYS"] - 1)
        fd = f - FPD * cfg["N_DAYS"] if in_live else f % FPD
        d = days[di_new]
        if di_new != di or f == 0:
            di = di_new
            d = days[di]
            xs_c, xs_p = xK(d["Kc"]), xK(d["Kp"])
            Vt = barrier(eng.x, xs_c, w_wall, d["Vc"]) + barrier(eng.x, xs_p, w_wall, d["Vp"])
            Vprev = Vt if Vday is None else Vday
            Vday = Vt
            eng.hbar = cfg["HBAR"] * d["a"] ** 2          # vol-scaled effective Planck constant
            if f == 0:
                eng.init_packet(xobs[0], cfg["SIGMA0"] * d["a"], d["p"]); pt = d["p"]; Vprev = Vday
            else:
                eng.measure(xobs[di], cfg["MEAS"] * d["a"])
                pnew = 0.35 * eng.mean_p() + 0.65 * d["p"]
                eng.kick(pnew - eng.mean_p(), xobs[di]); eng.normalise(); pt = pnew
            dirn = 1.0 if d["p"] >= 0 else -1.0
        if in_live and fd == 0:
            xs_c, xs_p = xK(live["Kc"]), xK(live["Kp"])
            Vprev = Vday
            Vday = barrier(eng.x, xs_c, w_wall, live["Vc"]) + barrier(eng.x, xs_p, w_wall, live["Vp"])
            d = dict(d, Kc=live["Kc"], Kp=live["Kp"], Vc=live["Vc"], Vp=live["Vp"], src="GLD OI / GEX (LIVE)")
            days[di] = d
            eng.measure(xobs[di], cfg["MEAS"] * d["a"])
            pn = 0.35 * eng.mean_p() + 0.65 * d["p"]
            eng.kick(pn - eng.mean_p(), xobs[di]); eng.normalise(); pt = pn
        if in_live: d = days[di]
        s = fd / 4.0 if not in_live else fd / 10.0
        s = np.clip(s, 0, 1); s = s * s * (3 - 2 * s)
        V = Vprev * (1 - s) + Vday * s
        for _ in range(SUB): eng.step(V, dt)
        xm = eng.mean_x(); pm = eng.mean_p()
        xs_c, xs_p = xK(d["Kc"]), xK(d["Kp"])
        dens = eng.dens(); dx = eng.dx
        Tu = float(np.sum(dens[eng.x > xs_c + w_wall / 2]) * dx)
        Td = float(np.sum(dens[eng.x < xs_p - w_wall / 2]) * dx)
        Tdir = Tu if dirn > 0 else Td
        V0d = d["Vc"] if dirn > 0 else d["Vp"]
        E = pt ** 2 / 2
        reg, rcol = regime(E, V0d, Tdir)
        xw, pg, Wd = wigner(eng, xm + dirn * 1.6)
        wscale = 1.0 / (np.pi * eng.hbar)
        # ---------- [1] outside 3D
        ax3.cla(); style3d(ax3)
        sel = (xw >= xm + dirn * 1.6 - 4.6) & (xw <= xm + dirn * 1.6 + 4.6)
        xw3, W3 = xw[sel], gaussian_filter(Wd[sel], (0.8, 1.0))
        sp = 2
        Xg, Pg = np.meshgrid(xw3, pg[::sp], indexing="ij")
        Zg = np.clip(W3[:, ::sp] / wscale, -0.5, 1.25)
        rgb = to_rgb(W3[:, ::sp], wscale)
        rgb = ls.shade_rgb(rgb, Zg, fraction=0.9, blend_mode="soft", vert_exag=1.0,
                           dx=xw3[1] - xw3[0], dy=pg[sp] - pg[0])
        ax3.plot_surface(Xg, Pg, Zg, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=True,
                         shade=False)
        x0, x1 = xw3[0], xw3[-1]; zt = 1.3; zb = -0.45
        for xs_, col in ((xs_c, AMBER), (xs_p, "#ffd27a")):
            if x0 < xs_ < x1:
                for xe in (xs_ - w_wall / 2, xs_ + w_wall / 2):
                    ax3.plot_surface(np.full((2, 2), xe), np.array([[-3.2, 3.2]] * 2),
                                     np.array([[zb, zb], [zt, zt]]), color=col, alpha=0.07, linewidth=0, shade=False)
                    for pp in np.linspace(-3.2, 3.2, 9):
                        ax3.plot([xe, xe], [pp, pp], [zb, zt], color=col, lw=0.7, alpha=0.5)
                    for zz in np.linspace(zb, zt, 6):
                        ax3.plot([xe, xe], [-3.2, 3.2], [zz, zz], color=col, lw=0.7, alpha=0.5)
        xm_ = np.linspace(x0, x1, 300)
        Vs = np.interp(xm_, eng.x, V)
        ax3.plot(xm_, np.full_like(xm_, 3.2), Vs / 1.6 + zb, color=AMBER, lw=2.2, alpha=0.95)
        trail.append((xm, pm)); trail[:] = trail[-60:]
        tr = np.array(trail)
        ax3.plot(tr[:, 0], tr[:, 1], np.full(len(tr), zb), color=CYAN, lw=1.3, alpha=0.75)
        ix = int(np.clip((xm - xw[0]) / (xw[1] - xw[0]), 0, len(xw) - 1)); ip = int(np.clip((pm - pg[0]) / (pg[1] - pg[0]), 0, len(pg) - 1))
        ax3.scatter([xm], [pm], [np.clip(Wd[ix, ip] / wscale, 0, 1.25) + 0.08], s=150, color="white",
                    edgecolors=CYAN, linewidths=2.2, depthshade=False)
        ax3.set_xlim(x0, x1); ax3.set_ylim(-3.2, 3.2); ax3.set_zlim(zb, zt)
        ax3.set_box_aspect((1.5, 1.0, 0.55), zoom=1.30)
        ph_ = f / max(NF - 1, 1)
        ax3.view_init(elev=25 - 5 * np.sin(ph_ * np.pi * 2), azim=-70 + 36 * ph_ + (14 if dirn < 0 else 0))
        ax3.grid(True)

        # ---------- [2] phase-space heatmap
        phase_trail.append((xm, pm))
        phase_trail[:] = phase_trail[-72:]
        path = np.asarray(phase_trail)
        phase_line.set_data(path[:, 0], path[:, 1])
        phase_point.set_offsets([[xm, pm]])
        im.set_data(np.clip(Wd / wscale, -0.35, 1.0).T)
        im.set_extent([xw[0], xw[-1], pg[0], pg[-1]])
        axt.set_xlim(xm + dirn * 1.6 - 4.6, xm + dirn * 1.6 + 4.6)
        axt.set_ylim(-2.8, 2.8)
        call_line.set_xdata([xs_c, xs_c])
        put_line.set_xdata([xs_p, xs_p])
        wall = "CALL WALL ▲" if dirn > 0 else "PUT WALL ▼"
        Kw = d["Kc"] if dirn > 0 else d["Kp"]
        hud1.set_text(f"{d['date']:%Y-%m-%d}   GC {d['S']:,.1f}" + (f"   ● LIVE · {LIVEF / FPD:.0f}d FORWARD" if in_live else ""))
        hud1.set_color("#19161a")
        hud2.set_text(f"{reg}")
        hud2.set_color("#275c53" if reg == "TUNNELING" else "#35437c" if reg == "EVANESCENT" else "#6b4b16")
        hud3.set_text(f"LOOKING {wall} {Kw:,.0f}\nT↑={Tu:.3f}  T↓={Td:.3f}  PIN={max(0, 1 - Tu - Td):.3f}\n"
                      f"E/V0={E / max(V0d, 1e-9):.2f}  V0={V0d:.2f}  ħ={eng.hbar:.2f}")

        # ---------- [3] time-by-signal heatmap
        n = cfg["N_DAYS"]; k = min(di + 1, n)
        visible = days[:k]
        momentum = 0.5 + 0.5 * np.tanh(np.array([q["p"] for q in visible]) / 1.25)
        vol_ratio = np.clip((np.array([q["a"] for q in visible]) - 0.6) / (1.7 - 0.6), 0, 1)
        wall_gaps = np.array([
            [max(0.0, (xK(q["Kc"]) - xobs[j]) / q["a"]),
             max(0.0, (xobs[j] - xK(q["Kp"])) / q["a"])]
            for j, q in enumerate(visible)
        ])
        call_proximity = np.exp(-wall_gaps[:, 0])
        put_proximity = np.exp(-wall_gaps[:, 1])
        heat = np.full((4, n), np.nan)
        heat[:, :k] = np.vstack((momentum, vol_ratio, call_proximity, put_proximity))
        market_img.set_data(np.ma.masked_invalid(heat))
        heat_cursor.set_xdata([di, di])
        tick_positions = np.linspace(0, n - 1, min(5, n), dtype=int)
        axc.set_xticks(tick_positions)
        axc.set_xticklabels([days[i]["date"].strftime("%m-%d") for i in tick_positions])
        axc.set_xlabel(f"trading day  ·  {d['date']:%Y-%m-%d}  ·  GC {d['S']:,.1f}",
                       color=DIM, fontsize=7, labelpad=3)
        sub.set_text(f"DATA {src}  |  BARRIERS {d['src']}  |  σ_d={d['sd'] * 100:.2f}%  |  {'live forward fan' if in_live else '1d horizon'} · split-step FFT")
        foot.set_text("Quantum-inspired state-space visualization; signals are illustrative, not calibrated forecasts or investment advice.")
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
        wr.add(frame)
        if f % 24 == 0:
            print(f"frame {f + 1}/{NF}   {time.time() - t0:5.0f}s")
    wr.close()
    print("saved ->", os.path.abspath(cfg["OUT"]), f"({time.time() - t0:.0f}s)")

main()
try:
    from IPython.display import Video, display
    display(Video(CFG["OUT"], embed=True, width=420))
except Exception:
    pass
