"""Generate the architecture figure of the feasibility prototype (DLT-Net) as SVG.

Style follows the project's Visio drawing template: Times New Roman, black 1px outlines,
pastel fills (#d8d8d8 #deebf6 #fff2cc #c8fdd3 #cbffff #ffcccc #fbe5d5 #e2efd9 #fee599),
accent lines #0070c0 / #c00000 / #375623 / #833c0b.
Kernel shapes in panels (b)/(c) use the same basis initialisation as proto.py.
"""
import math, random, sys, unicodedata
from xml.sax.saxutils import escape

W, H = 1600, 1060
FONT = "'Times New Roman', 'Liberation Serif', Times, serif"
BLUE, RED, GREEN, BROWN, GRAY = "#0070c0", "#c00000", "#375623", "#833c0b", "#7f7f7f"
out = []


def add(s):
    out.append(s)


# ------------------------------------------------------------------ rich text
def _italic_char(c):
    if unicodedata.category(c) == "Mn":
        return None  # combining mark: stays with previous run
    if "Α" <= c <= "Ω" or "℀" <= c <= "⅏":  # upright: capital Greek (Σ), ℝ
        return False
    return unicodedata.category(c).startswith("L")  # other letters (incl. Ŷ, lowercase Greek) italic


def rich(s, size):
    """Mini markup: $..$ math (letters italic), _{..} subscript, ^{..} superscript.
    Returns tspans; baseline is restored after every sub/sup."""
    spans, i, math_mode, level = [], 0, False, 0
    buf, cur_style = "", None

    def flush():
        nonlocal buf, cur_style
        if buf:
            spans.append((buf, cur_style, level))
        buf, cur_style = "", None

    stack = []
    while i < len(s):
        c = s[i]
        if c == "$":
            flush(); math_mode = not math_mode; i += 1; continue
        if c in "_^" and i + 1 < len(s) and s[i + 1] == "{":
            flush(); stack.append(level); level = -1 if c == "_" else 1; i += 2; continue
        if c == "}" and stack:
            flush(); level = stack.pop(); i += 1; continue
        it = _italic_char(c) if math_mode else False
        if it is None:
            buf += c; i += 1; continue
        style = "i" if it else "n"
        if cur_style is not None and style != cur_style:
            flush()
        cur_style = style; buf += c; i += 1
    flush()
    res, offset = [], 0.0
    for txt, style, lev in spans:
        fs = size if lev == 0 else size * 0.7
        target = {0: 0.0, -1: 0.28 * size, 1: -0.4 * size}[lev]
        dy = target - offset
        offset = target
        attrs = []
        if abs(dy) > 1e-6:
            attrs.append(f'dy="{dy:.2f}"')
        if lev != 0:
            attrs.append(f'font-size="{fs:.1f}"')
        if style == "i":
            attrs.append('font-style="italic"')
        res.append(f"<tspan {' '.join(attrs)}>{escape(txt)}</tspan>")
    if abs(offset) > 1e-6:
        res.append(f'<tspan dy="{-offset:.2f}">​</tspan>')
    return "".join(res)


def text(x, y, s, size=16, anchor="middle", weight=None, fill="#000", italic=False):
    w = f' font-weight="{weight}"' if weight else ""
    it = ' font-style="italic"' if italic else ""
    add(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" fill="{fill}"{w}{it}>{rich(s, size)}</text>')


# ------------------------------------------------------------------ shapes
def box(x, y, w, h, fill="#fff", stroke="#000", rx=6, dash=None, sw=1.2):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{d}/>')


def line(pts, arrow=True, dash=None, color="#000", sw=1.4):
    d = f' stroke-dasharray="{dash}"' if dash else ""
    m = ' marker-end="url(#arr)"' if arrow and color == "#000" else (f' marker-end="url(#arr_{color[1:]})"' if arrow else "")
    p = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
    add(f'<polyline points="{p}" fill="none" stroke="{color}" stroke-width="{sw}"{d}{m}/>')


def series(x0, x1, yc, amp, f, color, sw=1.6, dash=None, n=80):
    pts = [(x0 + (x1 - x0) * t / (n - 1), yc - amp * f(t / (n - 1))) for t in range(n)]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{color}" stroke-width="{sw}"{d} stroke-linejoin="round"/>')


def circle(cx, cy, r, fill="#fff", stroke="#000", sw=1.2):
    add(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


# ------------------------------------------------------------------ kernels (same init as proto.py)
L, R = 48, 4
delays = [L * 0.6 * r / (R - 1) for r in range(R)]
scales = [10 ** (math.log10(max(L / 4, 2)) * r / (R - 1)) for r in range(R)]


def phi(r, tau):
    d, s = delays[r], scales[r]
    return math.exp(-(tau - d) / s) if tau >= d else 0.0


def mini_axes(x0, y0, w, h, label_l=True, size=11):
    """axes with baseline at y0+h; returns mapping functions"""
    line([(x0, y0 + h), (x0 + w + 6, y0 + h)], arrow=False, color=GRAY, sw=1)
    line([(x0, y0 + h), (x0, y0 - 2)], arrow=False, color=GRAY, sw=1)
    if label_l:
        text(x0, y0 + h + 13, "0", size)
        text(x0 + w, y0 + h + 13, "$L$", size)
        text(x0 + w + 13, y0 + h + 4, "$τ$", size)
    return (lambda tau: x0 + w * tau / L), (lambda v: y0 + h - v)


def curve(fx, fy, g, color, sw=1.8, dash=None, n=97):
    pts = [(fx(L * i / (n - 1)), fy(g(L * i / (n - 1)))) for i in range(n)]
    d = f' stroke-dasharray="{dash}"' if dash else ""
    add(f'<polyline points="{" ".join(f"{a:.1f},{b:.1f}" for a, b in pts)}" fill="none" stroke="{color}" stroke-width="{sw}"{d} stroke-linejoin="round"/>')


def stems(fx, fy, taus, vals, color=BLUE):
    for t, v in zip(taus, vals):
        add(f'<line x1="{fx(t):.1f}" y1="{fy(0):.1f}" x2="{fx(t):.1f}" y2="{fy(v):.1f}" stroke="{color}" stroke-width="1.6"/>')
        circle(round(fx(t), 1), round(fy(v), 1), 2.4, fill=color, stroke=color, sw=0.8)


# ================================================================== document
add(f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" font-family="{FONT}">')
add("<defs>")
for name, col in [("arr", "#000"), ("arr_0070c0", BLUE), ("arr_c00000", RED), ("arr_7f7f7f", GRAY)]:
    add(f'<marker id="{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>')
add("</defs>")
add(f'<rect x="0" y="0" width="{W}" height="{H}" fill="#ffffff"/>')

rnd = random.Random(3)

# ---------------------------------------------------------------- (a) overall
# RevIN statistics skip
line([(250, 135), (250, 44), (1185, 44), (1185, 135)], dash="5,4", color=GRAY)
text(718, 36, "instance statistics $μ$, $σ$ (RevIN)", 14, fill="#404040")

# input
box(30, 105, 140, 130, "#ffffff")
base = lambda t: 0.55 * math.sin(2 * math.pi * 2.2 * t) + 0.3 * math.sin(2 * math.pi * 5.1 * t + 1)
series(40, 160, 130, 13, base, BLUE)
series(40, 160, 158, 13, lambda t: base(t - 0.07), RED)              # lagged copy -> lead-lag
series(40, 160, 186, 11, lambda t: 0.6 * math.sin(2 * math.pi * 3.3 * t + 2) + 0.2 * math.sin(2 * math.pi * 9 * t), GREEN)
series(40, 160, 213, 10, lambda t: base(t - 0.15) * 0.8 + 0.15 * math.sin(2 * math.pi * 11 * t), BROWN)
text(100, 258, "Input $X$ ∈ ℝ^{$C$×$T$}", 15)
text(100, 276, "($T$ = 96)", 13, fill="#404040")
line([(170, 170), (208, 170)])

box(210, 135, 80, 70, "#d8d8d8")
text(250, 175, "RevIN", 16)
line([(290, 170), (320, 170)], arrow=False)
line([(320, 90), (320, 250)], arrow=False)
line([(320, 90), (798, 90)])
line([(320, 250), (358, 250)])
text(560, 81, "$X̃$ ∈ ℝ^{$C$×$T$}", 15)

# intra-variable head
box(800, 60, 240, 60, "#deebf6")
text(920, 84, "Intra-variable head (CI)", 15, weight="bold")
text(920, 107, "shared Linear $W$_{self}: ℝ^{$T$} → ℝ^{$H$}", 14)

# DLT block
box(360, 200, 400, 100, "#fff2cc", sw=1.6)
text(560, 226, "Distributed-Lag Transfer (DLT)", 17, weight="bold")
text(560, 256, "$z_{i}(t)$ = Σ_{$j$∈$N$($i$)} Σ_{$τ$=0}^{$L$} $K_{ij}(τ)$ $x̃_{j}(t$ − $τ)$", 16)
text(560, 284, "one lag kernel per variable pair — detail in (b), variants in (c)", 12.5, fill="#404040")
line([(760, 250), (798, 250)])
text(779, 241, "$z$", 15)

# cross-variable head
box(800, 220, 240, 60, "#deebf6")
text(920, 244, "Cross-variable head", 15, weight="bold")
text(920, 267, "shared Linear $W$_{cross}: ℝ^{$T$} → ℝ^{$H$}", 14)

# sum
line([(1040, 90), (1090, 90), (1090, 152)])
line([(1040, 250), (1090, 250), (1090, 188)])
text(1100, 124, "$Ŷ$_{self}", 15, anchor="start")
text(1100, 228, "$Ŷ$_{cross}", 15, anchor="start")
circle(1090, 170, 16)
line([(1080, 170), (1100, 170)], arrow=False); line([(1090, 160), (1090, 180)], arrow=False)
line([(1106, 170), (1138, 170)])

box(1140, 135, 90, 70, "#d8d8d8")
text(1185, 175, "RevIN^{−1}", 16)
line([(1230, 170), (1268, 170)])

# output
box(1270, 105, 150, 130, "#ffffff")
split = 1345
for yc, amp, f, col in [(130, 13, base, BLUE), (158, 13, lambda t: base(t - 0.07), RED),
                        (186, 11, lambda t: 0.6 * math.sin(2 * math.pi * 3.3 * t + 2), GREEN),
                        (213, 10, lambda t: base(t - 0.15) * 0.8, BROWN)]:
    series(1280, split, yc, amp, lambda t, f=f: f(t * 0.55), "#a6a6a6", sw=1.3, n=45)
    series(split, 1410, yc, amp, lambda t, f=f: f(0.55 + t * 0.45), col, sw=2.0, n=40)
add(f'<line x1="{split}" y1="112" x2="{split}" y2="228" stroke="{GRAY}" stroke-width="1" stroke-dasharray="3,3"/>')
text(1345, 258, "Forecast $Ŷ$ ∈ ℝ^{$C$×$H$}", 15)
text(1345, 276, "($H$ ∈ {24, 96})", 13, fill="#404040")
line([(1420, 170), (1458, 170)])
box(1460, 145, 118, 50, "#ffcccc")
text(1519, 176, "MSE($Ŷ$, $Y$)", 16)

# offline neighbour selection
cx, ty = 100, 352
add(f'<path d="M{cx-55},{ty} v52 a55,11 0 0 0 110,0 v-52" fill="#e2efd9" stroke="#000" stroke-width="1.2"/>')
add(f'<ellipse cx="{cx}" cy="{ty}" rx="55" ry="11" fill="#e2efd9" stroke="#000" stroke-width="1.2"/>')
text(cx, ty + 30, "training", 14); text(cx, ty + 47, "split only", 14)
line([(155, 388), (358, 388)], dash="6,4")
box(360, 340, 400, 96, "#e2efd9", dash="6,4")
text(560, 364, "Neighbor &amp; lag selection (offline)".replace("&amp;", "&"), 15.5, weight="bold")
text(560, 390, "|CCF_{$ij$}($τ$)| on first differences, $τ$ = 0, …, $L$", 14)
text(560, 415, "→ top-$k$ sources $N$($i$) and peak lag $d_{ij}$", 14)
line([(560, 340), (560, 302)], dash="6,4")
text(570, 326, "$N$($i$), $d_{ij}$", 14, anchor="start")

# experimental setting box
box(830, 318, 748, 122, "#f2f2f2", stroke=GRAY, rx=4)
text(846, 342, "Experimental setting", 15, anchor="start", weight="bold")
text(846, 366, "Datasets: ETTh1, Weather, Electricity, PEMS08 (TSLib splits, z-score on train)", 14, anchor="start")
text(846, 389, "$T$ = 96,  $H$ ∈ {24, 96};   $k$ = 8 neighbors,  $L$ = 48 lags,  $R$ = 4 basis kernels", 14, anchor="start")
text(846, 412, "Adam (lr 3×10^{−3}, ×0.5 per epoch from epoch 2), ≤ 6 epochs, early stopping on val. MSE (patience 2)", 14, anchor="start")
text(846, 433, "All variants share $N$($i$), RevIN, $W$_{self}, $W$_{cross} — they differ only in how $K_{ij}$ is parameterized", 14, anchor="start")

text(800, 470, "(a) Overall architecture of the prototype", 18)

# ---------------------------------------------------------------- (b) DLT module
box(20, 492, 990, 528, "none", stroke=GRAY, rx=8, dash="4,4", sw=1)
# input icon
box(45, 600, 100, 80, "#ffffff")
series(53, 137, 618, 9, base, BLUE, sw=1.4)
series(53, 137, 640, 9, lambda t: base(t - 0.07), RED, sw=1.4)
series(53, 137, 662, 8, lambda t: base(t - 0.15) * 0.8, BROWN, sw=1.4)
text(95, 702, "$X̃$ ∈ ℝ^{$C$×$T$}", 14)
line([(145, 640), (183, 640)])

# step 1: basis filtering
box(185, 545, 245, 190, "#c8fdd3")
text(307, 569, "① Shared lag-basis filtering", 15, weight="bold")
cols = [(203, 583), (320, 583), (203, 640), (320, 640)]
for r, (x0, y0) in enumerate(cols):
    fx, fy = mini_axes(x0, y0, 92, 36, label_l=False)
    curve(fx, fy, lambda t, r=r: 33 * phi(r, t), [BLUE, RED, GREEN, BROWN][r], sw=1.6)
    text(x0 + 88, y0 + 10, f"$φ$_{{{r+1}}}", 13, anchor="end")
text(307, 701, "$u_{j,r}$ = $φ_{r}$ ∗ $x̃_{j}$,  $r$ = 1, …, $R$", 15)
text(307, 722, "(causal, $L$ + 1 taps, learnable)", 12.5, fill="#404040")
line([(430, 640), (468, 640)])
text(449, 631, "$U$", 14)

# step 2: neighbour gathering
box(470, 545, 200, 190, "#cbffff")
text(570, 569, "② Neighbor gathering", 15, weight="bold")
ti = (625, 638)
nbrs = [(515, 598, "$j$_{1}"), (515, 632, "$j$_{2}"), (515, 680, "$j_{k}$")]
for nx, ny, lab in nbrs:
    dx, dy = ti[0] - nx, ti[1] - ny
    dd = math.hypot(dx, dy)
    line([(nx + 12 * dx / dd, ny + 12 * dy / dd), (ti[0] - 16 * dx / dd, ti[1] - 16 * dy / dd)], color=BLUE, sw=1.3)
    circle(nx, ny, 12, "#ffffff")
    text(nx, ny + 5, lab, 13)
text(515, 660, "⋮", 15)
circle(ti[0], ti[1], 15, "#fff2cc")
text(ti[0], ti[1] + 5, "$i$", 15)
text(570, 711, "$U$[$N$($i$)] ∈ ℝ^{$C$×$k$×$R$×$T$}", 13.5)
line([(670, 640), (708, 640)])

# alpha parameter tag
box(770, 503, 150, 28, "#fee599", rx=4)
text(845, 522, "$α$ ∈ ℝ^{$C$×$k$×$R$} (learned)", 13.5)
line([(845, 531), (845, 543)])

# step 3: mixing
box(710, 545, 270, 190, "#fff2cc", sw=1.6)
text(845, 569, "③ Pair-wise lag mixing", 15, weight="bold")
text(845, 604, "$z_{i}$ = Σ_{$j$∈$N$($i$)} Σ_{$r$=1}^{$R$} $α_{ijr}$ $u_{j,r}$", 16)
text(845, 638, "⇔   $z_{i}$ = Σ_{$j$∈$N$($i$)} $K_{ij}$ ∗ $x̃_{j}$", 16)
text(845, 672, "$K_{ij}(τ)$ = Σ_{$r$} $α_{ijr}$ $φ_{r}(τ)$", 16)
text(845, 717, "output $z$ ∈ ℝ^{$C$×$T$} → $W$_{cross}", 13.5, fill="#404040")

# dynamic coefficients
box(470, 778, 510, 108, "#fbe5d5", dash="6,4")
text(725, 800, "Input-adaptive coefficients (DLT-dyn only)", 15, weight="bold")
text(725, 826, "$s_{i}$ = Linear_{$T$→16}($x̃_{i}$),   [$s_{i}$ ∥ $s_{j}$] → MLP (32 → $R$) → tanh", 14)
text(725, 853, "$α_{ijr}$($X$) = $α_{ijr}$ · (1 + tanh(·)),   last MLP layer zero-initialized", 14)
text(725, 876, "→ starts identical to static DLT, then modulates kernels per input window", 12.5, fill="#404040")
line([(845, 778), (845, 737)], dash="6,4", color=RED)
text(853, 763, "$α$($X$)", 14, anchor="start", fill=RED)
line([(95, 710), (95, 832), (468, 832)], dash="6,4", color=RED)

# complexity
box(45, 905, 935, 88, "#f2f2f2", stroke=GRAY, rx=4)
text(60, 930, "Cost per window:  filtering $O$($C$·$R$·$T$·$L$) + mixing $O$($C$·$k$·$R$·$T$)   vs.   free per-pair kernels $O$($C$·$k$·$T$·$L$)", 14.5, anchor="start")
text(60, 955, "Cross-kernel parameters:  $C$·$k$·$R$ + $R$($L$ + 1)   vs.   $C$·$k$·($L$ + 1)", 14.5, anchor="start")
text(60, 980, "e.g. PEMS08 ($C$ = 170): 5.6 K vs. 66.6 K;   Electricity ($C$ = 321): 10.5 K vs. 125.8 K", 14.5, anchor="start")
text(515, 1046, "(b) Distributed-Lag Transfer (DLT) module", 18)

# ---------------------------------------------------------------- (c) kernel variants
box(1030, 492, 550, 528, "none", stroke=GRAY, rx=8, dash="4,4", sw=1)
text(1305, 518, "Cross-variable kernel $K_{ij}(τ)$, $τ$ = 0, …, $L$", 15.5, weight="bold")
text(1305, 539, "baseline [ci]: no cross branch;   control [scalardyn]: [scalar] + input-adaptive $a_{ij}$($X$)", 12.5, fill="#404040")
cells = [
    ("Scalar mixing", "scalar", "$C$·$k$", "iTransformer / VCformer-style"),
    ("Single-lag shift", "shift", "$C$·$k$  ($d_{ij}$ fixed)", "LIFT-style"),
    ("Short kernel, 3 taps", "tap3", "3·$C$·$k$", "CrossLinear-style"),
    ("Free kernel", "full", "$C$·$k$·($L$ + 1)", "unconstrained"),
    ("DLT: shared lag basis", "dlt", "$C$·$k$·$R$ + $R$($L$ + 1)", "ours"),
    ("DLT-dyn: input-adaptive", "dltdyn", "DLT + small MLP", "ours"),
]
noise = [rnd.gauss(0, 1) for _ in range(L + 1)]
alpha = [0.30, 0.85, 0.55, 0.30]
alpha2 = [0.45, 0.50, 0.75, 0.45]
for n, (title, key, params, tag) in enumerate(cells):
    cx0 = 1045 + (n % 2) * 268
    cy0 = 553 + (n // 2) * 152
    ours = tag == "ours"
    box(cx0, cy0, 255, 142, "#fff2cc" if ours else "#ffffff", stroke="#000" if ours else "#a6a6a6", rx=5, sw=1.4 if ours else 1)
    text(cx0 + 10, cy0 + 20, f"{title}", 14, anchor="start", weight="bold")
    text(cx0 + 245, cy0 + 20, f"[{key}]", 12.5, anchor="end", fill=BLUE)
    fx, fy = mini_axes(cx0 + 18, cy0 + 35, 205, 55)
    if key == "scalar":
        stems(fx, fy, [0], [44])
    elif key == "shift":
        stems(fx, fy, [7], [44])
        text(fx(7), fy(48), "$d_{ij}$", 12)
    elif key == "tap3":
        stems(fx, fy, [0, 1, 2], [30, 44, 20])
    elif key == "full":
        vals = [max(-14, min(48, 38 * math.exp(-t / 9) + 8 * noise[t])) for t in range(L + 1)]
        stems(fx, fy, list(range(L + 1)), vals)
    elif key == "dlt":
        for r in range(R):
            curve(fx, fy, lambda t, r=r: 40 * alpha[r] * phi(r, t), "#a6a6a6", sw=1.1, dash="3,2")
        curve(fx, fy, lambda t: 40 * sum(alpha[r] * phi(r, t) for r in range(R)), BLUE, sw=2.2)
    elif key == "dltdyn":
        curve(fx, fy, lambda t: 40 * sum(alpha[r] * phi(r, t) for r in range(R)), BLUE, sw=2.2)
        curve(fx, fy, lambda t: 40 * sum(alpha2[r] * phi(r, t) for r in range(R)), RED, sw=2.0, dash="5,3")
        text(cx0 + 245, cy0 + 48, "input $X$", 11.5, anchor="end", fill=BLUE)
        text(cx0 + 245, cy0 + 62, "input $X$′", 11.5, anchor="end", fill=RED)
    text(cx0 + 10, cy0 + 133, f"params: {params}", 12.5, anchor="start")
    text(cx0 + 245, cy0 + 133, f"({tag})", 12, anchor="end", fill="#404040", italic=not ours)
text(1305, 1046, "(c) Kernel parameterizations compared", 18)

add("</svg>")
open(sys.argv[1], "w", encoding="utf-8").write("\n".join(out))
print("wrote", sys.argv[1])
