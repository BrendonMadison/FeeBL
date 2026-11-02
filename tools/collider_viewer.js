/** Viewer controller and particle rendering.
 * Reads authenticated local snapshots (um, urad, GeV), keeps frame/cache state,
 * and draws display samples. Display samples never define weighted observables.
 * Scientific renderers are loaded next; URL loading begins in the field module.
 */
const VARS = {
    z: {
        lab: "native display z  [µm]",
        get: p => p[3]
    },
    x: {
        lab: "x  [µm]",
        get: p => p[1]
    },
    y: {
        lab: "y  [µm]",
        get: p => p[2]
    },
    d: {
        lab: "δ  [10⁻³]",
        get: (p, e0) => 1e3 * (p[0] / e0 - 1)
    },
    r: {
        lab: "r = √(x²+y²)  [µm]",
        get: p => Math.hypot(p[1], p[2])
    },
    vx: {
        lab: "vx  [µrad]",
        get: p => p[4]
    },
    vy: {
        lab: "vy  [µrad]",
        get: p => p[5]
    },
    E: {
        lab: "energy  [GeV]",
        get: p => p[0]
    },
};
let files = {},
    man = null,
    deck = {},
    cache = new Map(),
    nfr = 0,
    e0 = [125, 125],
    scientific = null,
    diagnostics = null,
    nativeFields = null;
let show = [true, true],
    playing = false,
    timer = null,
    fixed = null;
// Dataset input is either a local folder or a loopback URL. Both routes use
// the same hashes, size limits, coordinate contracts and scientific validators.
let base = null,
    phCache = new Map(),
    stepL = [],
    win = null,
    lumiW = null,
    refW = null,
    refDir = null,
    pol = null;

function fail(e) {
    playing = false;
    clearInterval(timer);
    $("play").textContent = "Play";
    $("play").classList.remove("on");
    $("status").textContent = "Dataset error: " + e.message;
    $("status").style.color = "var(--warn)";
    $("drop").hidden = false;
    $("app").hidden = true;
}

function safeBase(value) {
    if (!/^[A-Za-z0-9_./-]+$/.test(value) || value.split("/").some(v => v === "..") || value
        .startsWith("/"))
        throw new Error("Run must be a local relative directory without traversal");
    if (!["localhost", "127.0.0.1", "[::1]"].includes(location.hostname))
        throw new Error("Serve datasets on loopback only");
    return value.replace(/\/$/, "");
}
/** Read one bounded local artifact as bytes; reject unsafe paths, redirects and oversized files. */
async function readBytes(name, limit = ColliderContract.LIMITS.fileBytes) {
    ColliderContract.safeName(name);
    if (base) {
        const r = await fetch(base + "/" + name, {
            redirect: "error"
        });
        if (!r.ok) throw new Error(name + ": HTTP " + r.status);
        if (Number(r.headers.get("content-length")) > limit) throw new Error("File size bound");
        const reader = r.body.getReader(),
            chunks = [];
        let total = 0;
        try {
            for (;;) {
                const {
                    done,
                    value
                } = await reader.read();
                if (done) break;
                total += value.length;
                if (total > limit) throw new Error("File size bound");
                chunks.push(value)
            }
        } finally {
            await reader.cancel()
        }
        const result = new Uint8Array(total);
        let at = 0;
        for (const c of chunks) {
            result.set(c, at);
            at += c.length
        }
        return result.buffer;
    }
    if (!files[name]) throw new Error(name + " missing");
    if (files[name].size > limit) throw new Error("File size bound");
    return files[name].arrayBuffer();
}
/** Decode bounded artifact bytes as strict UTF-8; malformed text throws. */
async function readText(name) {
    return new TextDecoder("utf-8", {
        fatal: true
    }).decode(await readBytes(name))
}

function has(name) {
    return base ? true : !!files[name];
}

const $ = id => document.getElementById(id);
for (const k in VARS) {
    for (const s of [$("ax"), $("ay")]) {
        const o = document.createElement("option");
        o.value = k;
        o.textContent = VARS[k].lab;
        s.appendChild(o);
    }
}
$("ax").value = "z";
$("ay").value = "x";

$("openAnother").addEventListener("click", () => $("dir").click());
$("dir").addEventListener("change", async ev => {
    if (!ev.target.files.length) return;
    const previous = {files, base, loaded: Boolean(man), frame: +$("fr").value};
    $("openAnother").disabled = true;
    try {
        const candidate = {};
        for (const file of ev.target.files) {
            if (candidate[file.name]) throw Error("Duplicate filename in selected folder: " + file.name);
            candidate[file.name] = file;
        }
        files = candidate;
        base = null;
        await load();
    } catch (error) {
        // Restore the last authenticated dataset if a replacement fails. File
        // objects retain their bytes even when the chooser selection changes.
        if (previous.loaded) {
            files = previous.files;
            base = previous.base;
            try {
                await load();
                $("fr").value = previous.frame;
                await draw();
                $("status").textContent = "New dataset rejected; previous run retained: " + error.message;
                $("status").style.color = "var(--warn)";
            } catch (restoreError) { fail(restoreError); }
        } else fail(error);
    } finally {
        $("openAnother").disabled = false;
        ev.target.value = "";
    }
});

function updateDatasetIdentity() {
    const relative = files["gui_manifest.json"]?.webkitRelativePath;
    const directory = base || (relative && relative.includes("/") ? relative.slice(0, relative
        .lastIndexOf("/")) : null);
    const sha = man.receipts?.card?.sha256;
    $("datasetIdentity").textContent = "Dataset: " + (directory ||
            "local folder (name unavailable)") + " · Input-card receipt SHA-256: " +
        (typeof sha === "string" && /^[a-f0-9]{64}$/.test(sha) ? sha.slice(0, 12) : "unavailable");
}
/** Authenticate and load one complete dataset; throws on invalid bytes or contracts. */
async function load() {
    playing = false;
    clearInterval(timer);
    $("play").textContent = "Play";
    $("play").classList.remove("on");
    $("status").textContent = "Loading manifest and checking every frame SHA256…";
    $("status").style.color = "";
    $("app").hidden = true;
    man = ColliderContract.manifest(JSON.parse(await readText("gui_manifest.json")));
    e0 = man.energy_GeV;
    nfr = man.frames.length;
    cache.clear();
    phCache.clear();
    fixed = null;
    win = null;
    deck = {};
    stepL = [];
    pol = null;
    lumiW = null;
    refW = null;
    for (const f of man.frames) {
        for (const k of ["beam1", "beam2"])
            await ColliderContract.artifact(await readBytes(f[k].path), f[k]);
        $("status").textContent = "Checked frame " + (f.id + 1) + " / " + nfr;
    }
    await loadScientific();
    updateDatasetIdentity();
    const backend = man.backend_actual_receipt.match(
        /compute_backend requested=\S+ actual=(\S+)/);
    $("receipt").textContent = "Source " + man.source.commit.slice(0, 12) + (man.source.dirty ?
            " + local changes" : "") +
        " · backend " + (backend ? backend[1] : "see runtime receipt") + " · validation: " + man
        .validation.status;
    $("runtimeReceipt").textContent = man.backend_actual_receipt;
    $("available").textContent = Object.entries(man.availability).map(([k, v]) => k.replaceAll(
        "_", " ") + ": " + v).join(" · ");
    $("sampling").textContent =
        "Native legacy collider dump coordinates; units µm, µrad, GeV. " +
        man.coordinates.qualification +
        " Deterministic sampling over both complete bunches; up to 4096 rows per beam per frame. " +
        "Unexported particles remain in the simulation. Samples are not weighted scientific observables.";
    $("status").textContent = "Integrity checked. Physics validation: " + man.validation
        .status +
        ". All frame hashes and finite particle rows matched.";
    { // fix the A_eff window at 4 sigma of frame 0
        const M0 = moments((await frame(0))[0], 0);
        if (M0) win = {
            x: 4 * Math.max(M0.sigx, 1e-6),
            y: 4 * Math.max(M0.sigy, 1e-6)
        };
    }
    $("fr").max = nfr - 1;
    $("fr").value = 0;
    $("drop").hidden = true;
    $("app").hidden = false;
    const O = optics();
    $("runinfo").textContent = nfr + " frames · native collider display · " + man.validation
        .status;
    const nativeRows = [
        ["Energy [GeV]", man.energy_GeV],
        ["IP dispersion [m]", man.ip_dispersion_m],
        ["crab_mode", man.crab_mode],
        ["waist_mode", man.waist_mode],
        ["native x angles [rad]", man.native_beam_angles_x_rad],
        ["native y angles [rad]", man.native_beam_angles_y_rad]
    ];
    $("optics").replaceChildren();
    for (const [name, value] of nativeRows) {
        const tr = document.createElement("tr"),
            key = document.createElement("td"),
            val = document.createElement("td");
        key.className = "k";
        key.textContent = name;
        val.textContent = Array.isArray(value) ? value.join(" / ") : "unavailable";
        tr.append(key, val);
        $("optics").append(tr);
    }
    $("narrow").textContent = "";
    await draw();
}

// What the optics are actually doing, from the deck plus the manifest.
// Deck units: beta in mm, normalised emittance in um, sigma_z in um, energy in
// GeV, angle_x is the HALF crossing angle.
function optics() {
    const D1 = (man.ip_dispersion_m || [0, 0])[0],
        D2 = (man.ip_dispersion_m || [0, 0])[1];
    const num = k => {
        const v = parseFloat(deck[k]);
        return isFinite(v) ? v : NaN;
    };
    const E = e0[0],
        gam = E / 0.510998950e-3;
    const emx = num("emitt_x"),
        bx = num("beta_x"),
        sz = num("sigma_z");
    const e1 = (man.espread || [0, 0])[0],
        e2 = (man.espread || [0, 0])[1];
    const sdeff = Math.sqrt((e1 * e1 + e2 * e2) / 2);
    const sbet = Math.sqrt((emx * 1e-6 / gam) * (bx * 1e-3)); // m
    const R = Math.abs(D1) * sdeff / sbet;
    const lam = Math.sqrt(1 + R * R);
    const stot = lam * sbet;
    const half = num("angle_x.1"),
        theta = 2 * half;
    const crab = (man.crab_mode || [0, 0])[0],
        waist = (man.waist_mode || [0, 0])[0];

    let mono, monoCls;
    if (!(Math.abs(D1) > 0)) {
        mono = "no dispersion";
        monoCls = "k";
    } else if (D1 * D2 < 0) {
        mono = "MONOCHROMATIZED";
        monoCls = "ok";
    } else {
        mono = "SAME-SIGN CONTROL";
        monoCls = "warn";
    }

    let scheme, schemeNote;
    if (!(theta > 0)) {
        scheme = "HEAD-ON";
        schemeNote = "no crossing angle";
    } else if (crab) {
        scheme = "CRAB CROSSING";
        schemeNote = "crab cancels the geometric angle";
    } else if (waist) {
        scheme = "CRAB WAIST";
        schemeNote = "no crab crossing: the sweep is not cancelled";
    } else {
        scheme = "UNCRABBED";
        schemeNote = "crossing angle with nothing compensating it";
    }

    const piw = stot > 0 ? sz * 1e-6 * Math.tan(half) / stot : NaN;
    return {
        D1,
        D2,
        sdeff,
        sbet,
        R,
        lam,
        stot,
        theta,
        half,
        crab,
        waist,
        mono,
        monoCls,
        scheme,
        schemeNote,
        piw,
        sz
    };
}

function parseDeck(t) {
    const o = {};
    const re = /([A-Za-z_][A-Za-z_0-9.]*)\s*=\s*([-0-9.eE+*a-z_]+)/g;
    let m;
    while ((m = re.exec(t))) o[m[1]] = m[2];
    return o;
}

function fmt(v, n) {
    if (!isFinite(v)) return "—";
    return Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(n) : v.toFixed(
    n);
}

/** Return validated particle rows for one frame; cache by frame ID. Rows use um, urad and GeV. */
async function frame(k) {
    if (cache.has(k)) return cache.get(k);
    const f = man.frames[k];
    if (!f) throw new Error("Frame outside manifest");
    const out = [];
    for (const name of ["beam1", "beam2"]) {
        const a = f[name];
        out.push(await ColliderContract.artifact(await readBytes(a.path), a));
    }
    if (cache.size >= 3) cache.delete(cache.keys().next().value);
    cache.set(k, out);
    return out;
}

// bp1.N / bp2.N are CUMULATIVE and append-ordered, so the photons radiated in
// frame k are the lines past the previous frame's count.
async function photons(k) {
    return [
        [],
        []
    ]
} // Not exported; never displayed as zero yield.

// out.dat carries one row per collision step: 2*n_z-1 of them, against
// frames*dump_step steps. Frame k is steps [k*dump_step, (k+1)*dump_step).
function parseStepLumi(t) {
    const i = t.indexOf("step   lumi total");
    if (i < 0) return [];
    const out = [];
    for (const ln of t.slice(i).split("\n").slice(1)) {
        const f = ln.trim().split(/\s+/);
        if (f.length === 3 && /^\d+$/.test(f[0])) out.push(+f[1]);
        else if (out.length) break;
    }
    return out;
}

// Effective area (sum rho)^2 / sum rho^2, as a histogram. The window is fixed
// at 4 sigma of frame 0 rather than hard-coded, so the pinch shows as a real
// change instead of being hidden by a window that shrinks with the beam.
function aEff(ps) {
    if (!win || !ps.length) return 0;
    const NB = 64,
        h = new Float64Array(NB * NB);
    let n = 0;
    for (const p of ps) {
        const x = p[1],
            y = p[2];
        if (x < -win.x || x >= win.x || y < -win.y || y >= win.y) continue;
        h[(((y + win.y) / (2 * win.y) * NB) | 0) * NB + (((x + win.x) / (2 * win.x) * NB) | 0)]++;
        n++;
    }
    if (!n) return 0;
    let s2 = 0;
    for (let i = 0; i < h.length; i++) {
        const p = h[i] / n;
        s2 += p * p;
    }
    return s2 > 0 ? ((2 * win.x / NB) * (2 * win.y / NB)) / s2 : 0;
}

// Energy lost since frame 0, split by where a particle sat in delta AT FRAME 0.
// Valid only because the dump order is stable: every particle's z shifts by the
// same amount between frames, so line i is the same particle throughout.
async function coreTail(k) {
    return null; // No stable particle-ID claim for display LOD.

}

// <E_gamma>/(Upsilon*E) is 4/(5*sqrt3) = 0.4619 only in the CLASSICAL limit.
// FeeBL samples the full quantum spectrum (physicalTools.cc:252), where photons
// are cut off at E_gamma <= E, so the ratio falls with Upsilon. These are
// numerical integrations of FeeBL's own kernel; using the classical constant
// instead underestimates Upsilon by 20-25% in the runs here.
const UPS_C = [
    [0.001, 0.4601],
    [0.010, 0.4406],
    [0.041, 0.3888],
    [0.058, 0.3672],
    [0.100, 0.3253],
    [0.244, 0.2426],
    [1.000, 0.1174]
];

function cOfUps(u) {
    if (u <= UPS_C[0][0]) return UPS_C[0][1];
    const L = UPS_C.length - 1;
    if (u >= UPS_C[L][0]) return UPS_C[L][1];
    for (let i = 1; i <= L; i++) {
        if (u <= UPS_C[i][0]) {
            const a = UPS_C[i - 1],
                b = UPS_C[i];
            const t = (Math.log(u) - Math.log(a[0])) / (Math.log(b[0]) - Math.log(a[0]));
            return a[1] + t * (b[1] - a[1]);
        }
    }
    return UPS_C[L][1];
}
// c(U)*U rises monotonically, so invert <E_gamma>/E = c(U)*U by bisection.
function upsFromEgamma(y) {
    if (!(y > 0)) return 0;
    // FCC-ee Z sits near 1e-6, so the bracket has to reach well below the
    // tabulated range; there c(U) is flat at its classical value anyway.
    let lo = 1e-9,
        hi = 5;
    for (let i = 0; i < 80; i++) {
        const m = Math.sqrt(lo * hi);
        if (cOfUps(m) * m < y) lo = m;
        else hi = m;
    }
    return Math.sqrt(lo * hi);
}

// Peak-region width, not rms. Same convention the study uses for sigma_W:
// smooth, find the peak, walk to half maximum each side, FWHM/2.3548. The rms
// of a post-collision energy distribution is dominated by the beamstrahlung
// tail and says almost nothing about where the collisions actually sit.
function coreOfHist(cen, wt) {
    const n = wt.length;
    if (n < 8) return null;
    const sm = wt.map((_, i) => (wt[Math.max(i - 1, 0)] + wt[i] + wt[Math.min(i + 1, n - 1)]) / 3);
    let pk = 0;
    for (let i = 1; i < n; i++)
        if (sm[i] > sm[pk]) pk = i;
    if (!(sm[pk] > 0)) return null;
    const half = sm[pk] / 2;
    let li = pk;
    while (li > 0 && sm[li] > half) li--;
    let ri = pk;
    while (ri < n - 1 && sm[ri] > half) ri++;
    let wl = cen[li],
        wr = cen[ri];
    if (li < pk && sm[li + 1] > sm[li]) wl = cen[li] + (cen[li + 1] - cen[li]) * (half - sm[li]) / (
        sm[li + 1] - sm[li]);
    if (ri > pk && sm[ri - 1] > sm[ri]) wr = cen[ri] + (cen[ri - 1] - cen[ri]) * (half - sm[ri]) / (
        sm[ri - 1] - sm[ri]);
    // a peak pinned against either edge gives a one-sided width; double the good half
    const okL = sm[li] <= half,
        okR = sm[ri] <= half;
    let fwhm;
    if (okL && okR) fwhm = wr - wl;
    else if (okR) fwhm = 2 * (wr - cen[pk]);
    else if (okL) fwhm = 2 * (cen[pk] - wl);
    else return null;
    // parabolic peak interpolation for the mode
    let mode = cen[pk];
    if (pk > 0 && pk < n - 1) {
        const a = sm[pk - 1],
            b2 = sm[pk],
            c = sm[pk + 1],
            d = a - 2 * b2 + c;
        if (d !== 0) mode = cen[pk] + 0.5 * (a - c) / d * (cen[1] - cen[0]);
    }
    return {
        mode,
        core: Math.max(fwhm, 0) / 2.3548,
        onesided: !(okL && okR)
    };
}

// Two passes, both windowed on a MAD scale rather than on percentiles. A
// percentile window is set by the radiative tail, which leaves too few bins
// across the peak and biases the width low; and with only ~4000 samples a fine
// binning lets the peak land on an upward fluctuation, which pushes the
// half-maximum crossings inward. Coarse pass locates the mode, fine pass
// measures the width around it.
function binIn(s, lo, hi, nb) {
    const wt = new Float64Array(nb),
        cen = new Array(nb);
    for (let i = 0; i < nb; i++) cen[i] = lo + (hi - lo) * (i + 0.5) / nb;
    for (let i = 0; i < s.length; i++) {
        const j = Math.floor((s[i] - lo) / (hi - lo) * nb);
        if (j >= 0 && j < nb) wt[j]++;
    }
    return {
        cen,
        wt: Array.from(wt)
    };
}

function coreOfSample(vals, nb) {
    const n = vals.length;
    if (n < 80) return null;
    const s = Float64Array.from(vals);
    s.sort();
    const med = s[n >> 1];
    const dev = Float64Array.from(s, v => Math.abs(v - med));
    dev.sort();
    const sig = 1.4826 * dev[n >> 1]; // robust sigma, tail-insensitive
    if (!(sig > 0)) return null;
    nb = nb || 64; // ~120 counts/bin at the peak
    let a = binIn(s, med - 5 * sig, med + 5 * sig, nb);
    let c0 = coreOfHist(a.cen, a.wt);
    if (!c0) return null;
    const b = binIn(s, c0.mode - 4 * sig, c0.mode + 4 * sig, nb);
    const c1 = coreOfHist(b.cen, b.wt) || c0;
    return gaussPeak(b.cen, b.wt, c1) || c1;
}

// A log-parabola fit over the peak region. For a Gaussian ln(counts) is exactly
// quadratic, so this measures the peak directly instead of inferring it from a
// half-maximum crossing -- which on ~4000 samples reads a few percent high
// because the smoothing that suppresses the peak-bin fluctuation also widens
// the flanks. Only bins above a quarter of the peak are used, so a radiative
// tail is excluded by construction rather than by windowing.
function gaussPeak(cen, wt, seed) {
    const n = wt.length;
    const sm = wt.map((_, i) => (wt[Math.max(i - 1, 0)] + wt[i] + wt[Math.min(i + 1, n - 1)]) / 3);
    let pk = 0;
    for (let i = 1; i < n; i++)
        if (sm[i] > sm[pk]) pk = i;
    const cut = 0.25 * sm[pk],
        x0 = cen[pk];
    let S = 0,
        Sx = 0,
        Sxx = 0,
        Sxxx = 0,
        Sxxxx = 0,
        Sy = 0,
        Sxy = 0,
        Sxxy = 0,
        used = 0;
    let xlo = Infinity,
        xhi = -Infinity;
    for (let i = 0; i < n; i++) {
        if (!(wt[i] > 0) || sm[i] < cut) continue;
        const x = cen[i] - x0,
            y = Math.log(wt[i]),
            w = wt[i]; // Poisson: var(ln w) ~ 1/w
        S += w;
        Sx += w * x;
        Sxx += w * x * x;
        Sxxx += w * x * x * x;
        Sxxxx += w * x * x * x * x;
        Sy += w * y;
        Sxy += w * x * y;
        Sxxy += w * x * x * y;
        used++;
        if (cen[i] < xlo) xlo = cen[i];
        if (cen[i] > xhi) xhi = cen[i];
    }
    if (used < 6) return null;
    // solve the 3x3 normal equations for y = a + b x + c x^2
    const M = [
            [S, Sx, Sxx],
            [Sx, Sxx, Sxxx],
            [Sxx, Sxxx, Sxxxx]
        ],
        v = [Sy, Sxy, Sxxy];
    for (let i = 0; i < 3; i++) {
        let p = i;
        for (let r = i + 1; r < 3; r++)
            if (Math.abs(M[r][i]) > Math.abs(M[p][i])) p = r;
        if (Math.abs(M[p][i]) < 1e-300) return null;
        [M[i], M[p]] = [M[p], M[i]];
        [v[i], v[p]] = [v[p], v[i]];
        for (let r = 0; r < 3; r++) {
            if (r === i) continue;
            const f = M[r][i] / M[i][i];
            for (let cc = i; cc < 3; cc++) M[r][cc] -= f * M[i][cc];
            v[r] -= f * v[i];
        }
    }
    const b = v[1] / M[1][1],
        c = v[2] / M[2][2];
    if (!(c < 0)) return null;
    const sd = Math.sqrt(-1 / (2 * c));
    const mode = x0 - b / (2 * c);
    // A skewed post-collision peak can send the vertex far outside the data and
    // the curvature to nearly zero, which reads as a huge width. Refuse anything
    // the fitted bins do not actually support and let the FWHM estimate stand.
    const span = xhi - xlo;
    if (!isFinite(sd) || sd <= 0 || !isFinite(mode)) return null;
    if (mode < xlo || mode > xhi) return null;
    if (sd > span || sd < span / (4 * n)) return null;
    return {
        mode,
        core: sd,
        onesided: seed ? seed.onesided : false,
        fit: true
    };
}

// sqrt(s) = 2 sqrt(E1 E2), exact head-on -- and FeeBL works in the head-on
// frame. Sampled pairings, not the true luminosity weighting: this is the
// collision energy AVAILABLE, not the one delivered. The luminosity-weighted
// answer is parsed separately from lumi.ee.out.dLdW.fne.
function comSample(a, b, K) {
    const n = Math.min(a.length, b.length);
    if (n < 80) return [];
    K = K || 24;
    const out = new Float64Array(n * K);
    let m = 0;
    for (let j = 0; j < K; j++) {
        const off = (j * 613 + 7) % n;
        for (let i = 0; i < n; i++) out[m++] = 2 * Math.sqrt(a[i][0] * b[(i * 7 + off) % n][0]);
    }
    return out;
}

// The luminosity spectrum FeeBL writes: this is the quantity the study calls
// sigma_W, so the viewer should agree with the analysis scripts exactly.
function parseDLdW(t) {
    const cen = [],
        wt = [];
    for (const ln of t.split("\n")) {
        if (!ln || ln[0] === "#") continue;
        const q = ln.trim().split(/\s+/);
        if (q.length >= 4) {
            const a = +q[0],
                b = +q[1],
                w = +q[3];
            if (isFinite(a) && isFinite(b) && isFinite(w)) {
                cen.push((a + b) / 2);
                wt.push(w);
            }
        }
    }
    return cen.length > 8 ? coreOfHist(cen, wt) : null;
}

// FeeBL already computes what is wanted here, in out.dat's "longitudinal bunch
// polarization" block: helicity at initial / collision / final, plus the
// luminosity-weighted pair. P is <s.p_hat>, the spin on each particle's own
// direction of motion, so beam 2 reads negative for a lab spin along +z. This
// parses that block rather than recomputing it from the lumi.ee.out spin
// columns, which would duplicate the source of truth.
function parsePol(t) {
    const num = "([-+]?[0-9.]+(?:e[-+]?[0-9]+)?)";
    const grab = (re) => {
        const m = t.match(re);
        return m ? m.slice(1).map(Number) : null;
    };
    const row = (word) => grab(new RegExp(
        "\\(P\\.1\\s*,\\s*P\\.2\\)\\s*at\\s*" + word + "\\s*=\\s*\\(\\s*" + num +
        "\\s*\\+-\\s*" + num + "\\s*,\\s*" + num + "\\s*\\+-\\s*" + num + "\\s*\\)" +
        "(?:\\s*\\[lab[^:]*:\\s*" + num + "\\s*,\\s*" + num + "\\s*\\])?"));
    const lw = grab(new RegExp(
        "\\(P\\.1\\s*,\\s*P\\.2\\)\\s*luminosity-weighted\\s*=\\s*\\(\\s*" + num +
        "\\s*,\\s*" + num + "\\s*\\)"));
    const shift = grab(new RegExp(
        "shift from initial, luminosity-weighted\\s*:\\s*beam 1\\s*" + num +
        "\\s*,\\s*beam 2\\s*" + num));
    const depol = grab(new RegExp(
        "depolarization \\(final - initial\\)\\s*:\\s*beam 1\\s*" + num +
        "\\s*,\\s*beam 2\\s*" + num));
    const ini = row("initial"),
        col = row("collision"),
        fin = row("final");
    if (!ini && !lw) return null;
    return {
        ini,
        col,
        fin,
        lw,
        shift,
        depol
    };
}

// Effective polarization for annihilation, in helicities:
//   P_eff = (P1 - P2) / (1 - P1 P2)
// FeeBL's P is already a helicity, so opposite-helicity beams give |P1 - P2|
// large and P_eff close to 1.
function pEff(p1, p2) {
    const d = 1.0 - p1 * p2;
    return d !== 0 ? (p1 - p2) / d : NaN;
}

/** Compute unweighted display-sample moments; never use these values as luminosity estimators. */
function moments(ps, b) {
    const n = ps.length;
    if (!n) return null;
    let sz = 0,
        sd = 0,
        szz = 0,
        sdd = 0,
        szd = 0,
        sx = 0,
        sxx = 0,
        sy = 0,
        syy = 0,
        se = 0;
    for (const p of ps) {
        const z = p[3],
            d = 1e3 * (p[0] / e0[b] - 1);
        sz += z;
        sd += d;
        szz += z * z;
        sdd += d * d;
        szd += z * d;
        sx += p[1];
        sxx += p[1] * p[1];
        sy += p[2];
        syy += p[2] * p[2];
        se += p[0];
    }
    const mz = sz / n,
        md = sd / n;
    const vz = szz / n - mz * mz,
        vd = sdd / n - md * md,
        cov = szd / n - mz * md;
    const h = vz > 0 ? cov / vz : 0; // delta = h z, per micron
    const r = vd > 0 ? (h * h * vz) / vd : 0; // correlated fraction
    return {
        n,
        sigz: Math.sqrt(Math.max(vz, 0)),
        sigd: Math.sqrt(Math.max(vd, 0)),
        sigx: Math.sqrt(Math.max(sxx / n - (sx / n) ** 2, 0)),
        sigy: Math.sqrt(Math.max(syy / n - (sy / n) ** 2, 0)),
        h,
        r: Math.max(0, Math.min(1, r)),
        mz,
        md,
        meanE: se / n,
        rho: (vz > 0 && vd > 0) ? cov / Math.sqrt(vz * vd) : 0
    };
}

function particleCanvasSize(canvas, context) {
    // Use CSS pixel coordinates and a matching backing store: resizing the
    // canvas must not independently squeeze the particle or field projection.
    if (typeof innerHeight !== "number" || !canvas.getBoundingClientRect) return {
        width: canvas.width,
        height: canvas.height
    };
    const box = canvas.getBoundingClientRect(),
        bar = $("playback").getBoundingClientRect();
    const top = box.top + (typeof scrollY === "number" ? scrollY : 0);
    const width = Math.max(240, Math.round(box.width)),
        height = Math.max(240, Math.min(660, Math.floor(innerHeight - top - bar.height - 16)));
    const ratio = typeof devicePixelRatio === "number" ? Math.max(1, Math.min(3,
        devicePixelRatio)) : 1;
    canvas.style.height = height + "px";
    const bw = Math.round(width * ratio),
        bh = Math.round(height * ratio);
    if (canvas.width !== bw || canvas.height !== bh) {
        canvas.width = bw;
        canvas.height = bh
    }
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return {
        width,
        height
    };
}
let drawBusy = false,
    drawPending = false;
/** Serialize asynchronous rendering so stale frame requests do not overwrite newer selections. */
async function draw() {
    drawPending = true;
    if (drawBusy) return;
    drawBusy = true;
    try {
        do {
            drawPending = false;
            await renderFrame()
        } while (drawPending);
    } catch (e) {
        drawPending = false;
        fail(e)
    } finally {
        drawBusy = false
    }
}
/** Draw one selected frame with fixed run bounds and native sample coordinates. */
async function renderFrame() {
    const k = +$("fr").value;
    updateProjectionContext();
    $("frlab").textContent = k + " / " + (nfr - 1) + " · t=" + (man.frames[k].elapsed_s * 1e12)
        .toPrecision(4) + " ps since first diagonal";
    const F = await frame(k);
    const gx = VARS[$("ax").value],
        gy = VARS[$("ay").value];
    const c = $("c"),
        g = c.getContext("2d"),
        size = particleCanvasSize(c, g);
    const cs = getComputedStyle(document.documentElement);
    const col = n => cs.getPropertyValue(n).trim();
    g.clearRect(0, 0, size.width, size.height);
    g.fillStyle = col("--panel");
    g.fillRect(0, 0, size.width, size.height);

    const L = 76,
        R = 26,
        T = 30,
        B = 54,
        W = size.width - L - R,
        H = size.height - T - B;

    // Full view uses every recorded bound across both bunches, including asymmetric tails.
    function allBounds(axis) {
        const col = {
            E: 0,
            x: 1,
            y: 2,
            z: 3,
            vx: 4,
            vy: 5
        } [axis];
        let lo = Infinity,
            hi = -Infinity;
        for (const f of man.frames)
            for (let b = 0; b < 2; b++) {
                const a = f[b === 0 ? "beam1" : "beam2"];
                let range = col === undefined ? null : a.bounds[col];
                if (axis === "d") range = a.bounds[0].map(e => 1e3 * (e / e0[b] - 1));
                if (axis === "r") range = [0, Math.hypot(Math.max(...a.bounds[1].map(Math.abs)),
                    Math.max(...a.bounds[2].map(Math.abs)))];
                lo = Math.min(lo, range[0]);
                hi = Math.max(hi, range[1]);
            }
        return [lo, hi];
    }

    function central(g) {
        const v = F.flatMap((p, b) => p.map(q => g.get(q, e0[b])));
        const mean = v.reduce((x, y) => x + y, 0) / v.length;
        const sd = Math.sqrt(v.reduce((x, y) => x + (y - mean) ** 2, 0) / v.length);
        return [mean - 3 * sd, mean + 3 * sd];
    }
    const pad = a => {
        const d = (a[1] - a[0]) || 1;
        return [a[0] - .06 * d, a[1] + .06 * d]
    };
    const fieldWindow = ColliderFields.samplingWindow(nativeFields, ColliderFields.viewKey($(
        "ax").value, $("ay").value), selectedFieldData());
    const fieldCrop = $("view").value === "field" && fieldWindow;
    const includeField = $("fieldMode").value === "on" && fieldWindow;
    const union = (a, b) => b ? [Math.min(a[0], b[0]), Math.max(a[1], b[1])] : a;
    const bx = pad(fieldCrop ? fieldWindow.horizontal : $("view").value === "central" ? central(
        gx) : union(allBounds($("ax").value), includeField && fieldWindow.horizontal));
    const by = pad(fieldCrop ? fieldWindow.vertical : $("view").value === "central" ? central(
        gy) : union(allBounds($("ay").value), includeField && fieldWindow.vertical));
    const px = v => L + (v - bx[0]) / (bx[1] - bx[0]) * W;
    const py = v => T + H - (v - by[0]) / (by[1] - by[0]) * H;

    paintFieldOverlay(k, g, {
        L,
        T,
        W,
        H,
        px,
        py,
        horizontal: $("ax").value,
        vertical: $("ay").value
    });

    // grid + frame
    g.strokeStyle = col("--line");
    g.lineWidth = 1;
    g.font = "12px " + cs.getPropertyValue("--mono");
    g.fillStyle = col("--mid");
    g.textAlign = "center";
    g.textBaseline = "top";
    const horizontalTicks = ColliderViewerUI.linearTicks(bx[0], bx[1], W);
    for (const v of horizontalTicks.values) {
        const X = px(v);
        g.beginPath();
        g.moveTo(X, T);
        g.lineTo(X, T + H);
        g.stroke();
        g.fillText(ColliderViewerUI.tickLabel(v, horizontalTicks.step), X, T + H + 8);
    }
    g.textAlign = "right";
    g.textBaseline = "middle";
    const verticalTicks = ColliderViewerUI.linearTicks(by[0], by[1], H);
    for (const v of verticalTicks.values) {
        const Y = py(v);
        g.beginPath();
        g.moveTo(L, Y);
        g.lineTo(L + W, Y);
        g.stroke();
        g.fillText(ColliderViewerUI.tickLabel(v, verticalTicks.step), L - 8, Y);
    }
    g.strokeStyle = col("--mid");
    g.strokeRect(L, T, W, H);
    g.fillStyle = col("--ink");
    g.textAlign = "center";
    g.textBaseline = "top";
    g.font = "13px " + cs.getPropertyValue("--sans");
    g.fillText(gx.lab, L + W / 2, T + H + 26);
    g.save();
    g.translate(20, T + H / 2);
    g.rotate(-Math.PI / 2);
    g.fillText(gy.lab, 0, 0);
    g.restore();

    // points
    const cols = [col("--e1"), col("--e2")];
    for (let b = 0; b < 2; b++) {
        if (!show[b]) continue;
        g.fillStyle = cols[b];
        g.globalAlpha = 0.55;
        for (const p of F[b]) {
            const X = px(gx.get(p, e0[b])),
                Y = py(gy.get(p, e0[b]));
            if (X < L || X > L + W || Y < T || Y > T + H) continue;
            if (b === 0) g.fillRect(X - 1, Y - 1, 2, 2);
            else { g.beginPath(); g.moveTo(X, Y - 2); g.lineTo(X + 2, Y);
                g.lineTo(X, Y + 2); g.lineTo(X - 2, Y); g.closePath(); g.fill(); }
        }
        g.globalAlpha = 1;
    }

    // chirp fit line, only meaningful on z-delta
    const M = [moments(F[0], 0), moments(F[1], 1)];
    if ($("fit").classList.contains("on") &&
        $("ax").value === "z" && $("ay").value === "d") {
        for (let b = 0; b < 2; b++) {
            if (!show[b] || !M[b]) continue;
            g.strokeStyle = cols[b];
            g.lineWidth = 2;
            g.setLineDash([7, 4]);
            g.beginPath();
            g.moveTo(px(bx[0]), py(M[b].md + M[b].h * (bx[0] - M[b].mz)));
            g.lineTo(px(bx[1]), py(M[b].md + M[b].h * (bx[1] - M[b].mz)));
            g.stroke();
            g.setLineDash([]);
        }
    }

    // stats
    const rows = [];
    const nm = ["beam 1 display sample", "beam 2 display sample"];
    const esp = man.espread || [0, 0];
    const Wdes = 2 * Math.sqrt(e0[0] * e0[1]);

    rows.push(['<td class="k" colspan="2"><b>design (deck)</b></td>']);
    rows.push(['<td class="k">E1 / E2</td><td>' + fmt(e0[0], 4) + " / " + fmt(e0[1], 4) +
        " GeV</td>"
    ]);
    rows.push(['<td class="k">σδ1 / σδ2</td><td>' + fmt(esp[0] * 1e3, 3) + " / " + fmt(esp[1] *
        1e3, 3) + "e−3</td>"]);
    rows.push(['<td class="k">design nominal W</td><td>' + (scientific ? fmt(scientific
            .definition.nominal_W_GeV, 5) + " GeV" :
        "unavailable: explicit design angle required") + '</td>']);

    for (let b = 0; b < 2; b++) {
        if (!M[b]) continue;

        rows.push(['<td class="' + (b ? "b2" : "b1") + '" colspan="2"><b>' + nm[b] +
            "</b></td>"]);
        rows.push(['<td class="k">particles</td><td>' + M[b].n + "</td>"]);
        rows.push(['<td class="k">σx</td><td>' + fmt(M[b].sigx * 1e3, 1) + " nm</td>"]);
        rows.push(['<td class="k">σy</td><td>' + fmt(M[b].sigy * 1e3, 1) + " nm</td>"]);
        rows.push(['<td class="k">σz</td><td>' + fmt(M[b].sigz, 1) + " µm</td>"]);
        rows.push(['<td class="k">σδ rms</td><td>' + fmt(M[b].sigd, 3) + "e−3</td>"]);
        rows.push(['<td class="k">h</td><td>' + fmt(M[b].h * 1e3, 3) + " /m</td>"]);
        rows.push(['<td class="k">ρ(z,δ)</td><td>' + fmt(M[b].rho, 3) + "</td>"]);
    }

    rows.push(['<td class="k" colspan="2">' + (scientific ? (monochromatization?.peak_fit ?
                "Native e+e− spectrum and spectrum-model shape analysis below." :
                "Native e+e− spectrum shown below; Gaussian peak fit unavailable.") :
            "e+e− invariant-mass spectrum unavailable; no random pairing estimator.") +
        '</td>'
    ]);
    $("sampleStats").innerHTML = rows.map(r => "<tr>" + r + "</tr>").join("");
    $("r1").textContent = M[0] ? fmt(M[0].r, 3) : "—";
    $("r2").textContent = M[1] ? fmt(M[1].r, 3) : "—";

    drawScientificViews(k);

}

$("fr").addEventListener("input", draw);
$("ax").addEventListener("change", () => {
    fixed = null;
    draw()
});
$("ay").addEventListener("change", () => {
    fixed = null;
    draw()
});
for (const [id, i] of [
        ["t1", 0],
        ["t2", 1]
    ])
    $(id).addEventListener("click", e => {
        show[i] = !show[i];
        e.target.classList.toggle("on", show[i]);
        draw()
    });
$("fit").addEventListener("click", e => {
    e.target.classList.toggle("on");
    draw()
});
$("lock").addEventListener("click", e => {
    e.target.classList.toggle("on");
    fixed = null;
    draw()
});
document.querySelectorAll(".presets button").forEach(b => {
    b.addEventListener("click", () => {
        $("ax").value = b.dataset.x;
        $("ay").value = b.dataset.y;
        fixed = null;
        draw();
    });
});
$("play").addEventListener("click", () => {
    playing = !playing;
    $("play").textContent = playing ? "Pause" : "Play";
    $("play").classList.toggle("on", playing);
    if (playing) {
        timer = setInterval(() => {
            let v = (+$("fr").value + 1) % nfr;
            $("fr").value = v;
            draw();
        }, +$("speed").value);
    } else clearInterval(timer);
});
addEventListener("keydown", e => {
    if (!man || !ColliderViewerUI.acceptsFrameShortcut(e)) return;
    e.preventDefault();
    if (e.key === "ArrowRight") {
        $("fr").value = Math.min(nfr - 1, +$("fr").value + 1);
        draw()
    }
    if (e.key === "ArrowLeft") {
        $("fr").value = Math.max(0, +$("fr").value - 1);
        draw()
    }
});

function updateProjectionContext() {
    const x = $("ax").value,
        y = $("ay").value,
        transverse = (x === "x" && y === "y") || (x === "y" && y === "x");
    const window = ColliderFields.samplingWindow(nativeFields, ColliderFields.viewKey(x, y),
        selectedFieldData());
    const fieldCrop = $("view").value === "field" && window,
        fieldVisible = $("fieldMode").value === "on" && window;
    const extent = fieldCrop ? "fixed field sampling window; particles outside it are cropped" : $(
            "view").value === "central" ? "central ±3σ crop of this frame" : fieldVisible ?
        "complete bunches and field window, fixed across frames" :
        "complete sampled bunches, fixed across frames";
    const meaning = transverse ?
        "Looking along the collision axis: longitudinal separation is hidden." :
        (x === "z" || y === "z") ? "The z axis shows longitudinal approach and separation." :
        "This projection does not show longitudinal separation.";
    $("projectionContext").textContent = x + "–" + y + " projection · " + extent + ". " + meaning +
        (fieldVisible ?
            " Dashed outline: sampling extent across the run. Blank regions are unsampled, not zero field." :
            " Field z–x and Field z–y keep z horizontal and preserve the frame.");
}

function setParticleProjection(view) {
    $("ax").value = "z";
    $("ay").value = view === "zy" ? "y" : "x";
    if (view === "overview") $("view").value = "full";
    else {
        const available = Boolean(selectedFieldLayer()?.availableFrames);
        $("fieldMode").value = available ? "on" : "off";
        if (!available) $("fieldNote").textContent = "Longitudinal field overlay unavailable: " +
            (nativeFields?.data.frames[+$("fr").value]?.[ColliderFields.dataMode(nativeFields,
                    selectedFieldData()) === "snapshot" ? "snapshot" : "longitudinal"]?.reason ||
                "this dataset has no authenticated longitudinal field export.");
    }
    fixed = null;
    updateProjectionContext();
    draw();
}
$("collisionOverview").addEventListener("click", () => setParticleProjection("overview"));
$("fieldZX").addEventListener("click", () => setParticleProjection("zx"));
$("fieldZY").addEventListener("click", () => setParticleProjection("zy"));
$("fitField").addEventListener("click", () => {
    if (!["zx", "zy"].includes(ColliderFields.viewKey($("ax").value, $("ay").value))) {
        $("ax").value = "z";
        $("ay").value = "x"
    }
    if (ColliderFields.samplingWindow(nativeFields, ColliderFields.viewKey($("ax").value, $(
            "ay").value), selectedFieldData())) {
        $("view").value = "field";
        $("fieldMode").value = "on";
    }
    draw();
});
$("view").addEventListener("change", draw);
$("reset").addEventListener("click", () => {
    $("view").value = "full";
    $("ax").value = "z";
    $("ay").value = "x";
    $("fr").value = 0;
    draw();
});
$("speed").addEventListener("change", () => {
    if (playing) {
        $("play").click();
        $("play").click()
    }
});

function download(name, text, type) {
    const url = URL.createObjectURL(new Blob([text], {
        type
    }));
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}
$("manifestDownload").addEventListener("click", () => download("gui_manifest.json", JSON.stringify(
    man, null, 2), "application/json"));
$("frameDownload").addEventListener("click", async () => {
    try {
        const k = +$("fr").value,
            F = await frame(k);
        const lines = ["beam,display_row,energy_GeV,x_um,y_um,z_um,vx_urad,vy_urad"];
        F.forEach((beam, b) => beam.forEach((row, i) => lines.push([b + 1, i, ...row
            .slice(0, 6)
        ].join(","))));
        download("display_frame_" + k + ".csv", lines.join("\n"), "text/csv");
    } catch (e) {
        fail(e)
    }
});
