"""Numba versions of three Monte Carlo kernels, for the speed comparison in step 7.

NumPy is fast when a computation is one big vectorised expression. It struggles in two ways:
it writes every intermediate array to memory (a 252-step path set is 252 times the data of a
terminal draw), and it can't branch per element without computing both sides under masks.
Numba compiles a plain per-path loop to machine code, so each path lives in registers and takes
only the branch it needs.

Random numbers: xoshiro256** (Blackman and Vigna) with Box–Muller, written out in the kernel
so the generator state lives in local variables. Numba's built-in np.random is the obvious
choice, but in a parallel loop every draw has to look up the current thread's generator. That
made the parallel build 2.3x slower per thread than the serial one, which cancelled the gain
from a second thread. Each block of ``BLOCK`` paths seeds its own state from (seed, block) via
splitmix64. A block always runs on one thread from start to finish, so results are identical
whatever the thread count: serial and parallel runs agree to the last bit. (It is a different
stream from NumPy's, so prices match NumPy statistically, not bit for bit.)

The ``*_from_normals`` variants take the random numbers as arrays, which allows an exact
comparison with the NumPy code paths on identical inputs.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from mc_engine.pricer import MCResult, Seed, _RunningMoments

BLOCK = 4096
PSI_CRIT = 1.5
SQRT1_2 = 1.0 / math.sqrt(2.0)


U64 = np.uint64
TWO_PI = 2.0 * math.pi


def _block_seed(seed: Seed) -> np.uint64:
    """64-bit kernel seed. Always passed as np.uint64: a plain Python int would be typed int64
    or uint64 depending on its size, and a kernel compiled for one rejects the other."""
    ss = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    return ss.generate_state(1, dtype=np.uint64)[0]


def _rotl(x, k):
    return (x << U64(k)) | (x >> U64(64 - k))


def _splitmix(x):
    x = x + U64(0x9E3779B97F4A7C15)
    z = (x ^ (x >> U64(30))) * U64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> U64(27))) * U64(0x94D049BB133111EB)
    return x, z ^ (z >> U64(31))


def _seed_state(seed, block):
    x = U64(seed) ^ (U64(block) * U64(0xD1B54A32D192ED03))
    x, a = _splitmix(x)
    x, b = _splitmix(x)
    x, c = _splitmix(x)
    x, d = _splitmix(x)
    return a, b, c, d


def _uniform(s0, s1, s2, s3):
    """xoshiro256**: one uniform in (0, 1), never exactly 0, plus the new state."""
    result = _rotl(s1 * U64(5), 7) * U64(9)
    t = s1 << U64(17)
    s2 ^= s0
    s3 ^= s1
    s1 ^= s2
    s0 ^= s3
    s2 ^= t
    s3 = _rotl(s3, 45)
    return ((result >> U64(11)) + 0.5) * (1.0 / 9007199254740992.0), s0, s1, s2, s3


def _normal(s0, s1, s2, s3, has_spare, spare):
    """Box–Muller, returning the second normal of each pair on the next call."""
    if has_spare:
        return spare, s0, s1, s2, s3, False, 0.0
    u1, s0, s1, s2, s3 = _uniform(s0, s1, s2, s3)
    u2, s0, s1, s2, s3 = _uniform(s0, s1, s2, s3)
    r = math.sqrt(-2.0 * math.log(u1))
    return r * math.cos(TWO_PI * u2), s0, s1, s2, s3, True, r * math.sin(TWO_PI * u2)


def _combine(blocks: np.ndarray, n_paths: int, method: str) -> MCResult:
    """Merge per-block (count, mean, M2) rows with Chan's parallel update, in block order."""
    acc = _RunningMoments()
    for nb, mb, qb in blocks:
        if nb == 0:
            continue
        tot = acc.n + nb
        delta = mb - acc.mean
        acc.mean += delta * nb / tot
        acc.m2 += qb + delta**2 * acc.n * nb / tot
        acc.n = tot
    return MCResult(float(acc.mean), acc.std_error(), n_paths, method=method)


# Pattern: each kernel = a small parallel driver looping over blocks + a per-block function
# compiled on its own. Writing the whole loop body inside prange was 2.1x slower per thread
# (Numba's parallel transform outlines that body and LLVM then optimises it less well), which
# cancelled the gain from a second thread. Calling a normally compiled function from a trivial
# prange loop keeps full single-thread speed.


def _european_block(s0, k, drift, vol, disc, is_call, start, end, seed, b):
    r0, r1, r2, r3 = _seed_state(seed, b)
    has, spare = False, 0.0
    mean, m2, count = 0.0, 0.0, 0.0
    for _ in range(start, end):
        z, r0, r1, r2, r3, has, spare = _normal(r0, r1, r2, r3, has, spare)
        st = s0 * math.exp(drift + vol * z)
        pay = disc * (max(st - k, 0.0) if is_call else max(k - st, 0.0))
        count += 1.0
        d = pay - mean
        mean += d / count
        m2 += d * (pay - mean)
    return count, mean, m2


def _asian_block(s0, k, drift, vol, disc, is_call, n_steps, start, end, seed, b):
    r0, r1, r2, r3 = _seed_state(seed, b)
    has, spare = False, 0.0
    mean, m2, count = 0.0, 0.0, 0.0
    log_s0 = math.log(s0)
    for _ in range(start, end):
        log_s = log_s0
        total = 0.0
        for _ in range(n_steps):
            z, r0, r1, r2, r3, has, spare = _normal(r0, r1, r2, r3, has, spare)
            log_s += drift + vol * z
            total += math.exp(log_s)
        avg = total / n_steps
        pay = disc * (max(avg - k, 0.0) if is_call else max(k - avg, 0.0))
        count += 1.0
        d = pay - mean
        mean += d / count
        m2 += d * (pay - mean)
    return count, mean, m2


# ------------------------------------------------------------------------- Heston QE (call)


def _qe_step(v, x, zv, zx, theta, xi, kappa, ek, k0, k1, k2, k3, k4, drift_dt):
    """One Andersen QE step for a single path. Returns (v_new, x_new)."""
    m = theta + (v - theta) * ek
    s2 = v * xi * xi * ek * (1.0 - ek) / kappa + theta * xi * xi * (1.0 - ek) * (1.0 - ek) / (
        2.0 * kappa
    )
    psi = s2 / (m * m)
    if psi <= PSI_CRIT:
        inv = 2.0 / psi
        b2 = inv - 1.0 + math.sqrt(inv) * math.sqrt(inv - 1.0)
        a = m / (1.0 + b2)
        root = math.sqrt(b2) + zv
        v_new = a * root * root
    else:
        p0 = (psi - 1.0) / (psi + 1.0)
        beta = (1.0 - p0) / m
        u = 0.5 * math.erfc(-zv * SQRT1_2)
        one_minus_u = 0.5 * math.erfc(zv * SQRT1_2)
        v_new = 0.0 if u <= p0 else math.log((1.0 - p0) / max(one_minus_u, 1e-300)) / beta
    x_new = x + drift_dt + k0 + k1 * v + k2 * v_new + math.sqrt(k3 * v + k4 * v_new) * zx
    return v_new, x_new


def _qe_constants(t, n_steps, kappa, theta, xi, rho):
    dt = t / n_steps
    ek = math.exp(-kappa * dt)
    k0 = -rho * kappa * theta * dt / xi
    k1 = 0.5 * dt * (kappa * rho / xi - 0.5) - rho / xi
    k2 = 0.5 * dt * (kappa * rho / xi - 0.5) + rho / xi
    k3 = 0.5 * dt * (1.0 - rho * rho)
    return dt, ek, k0, k1, k2, k3, k3


def _heston_block(
    s0,
    k,
    disc,
    is_call,
    n_steps,
    v0,
    theta,
    xi,
    kappa,
    ek,
    k0,
    k1,
    k2,
    k3,
    k4,
    drift_dt,
    start,
    end,
    seed,
    b,
):
    r0, r1, r2, r3 = _seed_state(seed, b)
    has, spare = False, 0.0
    mean, m2, count = 0.0, 0.0, 0.0
    log_s0 = math.log(s0)
    for _ in range(start, end):
        v = v0
        x = log_s0
        for _ in range(n_steps):
            zv, r0, r1, r2, r3, has, spare = _normal(r0, r1, r2, r3, has, spare)
            zx, r0, r1, r2, r3, has, spare = _normal(r0, r1, r2, r3, has, spare)
            v, x = _qe_step(v, x, zv, zx, theta, xi, kappa, ek, k0, k1, k2, k3, k4, drift_dt)
        st = math.exp(x)
        pay = disc * (max(st - k, 0.0) if is_call else max(k - st, 0.0))
        count += 1.0
        d = pay - mean
        mean += d / count
        m2 += d * (pay - mean)
    return count, mean, m2


# ----------------------------------------------------------------------- parallel drivers


def _european_blocks(s0, k, t, r, sigma, q, is_call, n_paths, seed):
    n_blocks = (n_paths + BLOCK - 1) // BLOCK
    out = np.zeros((n_blocks, 3))
    drift = (r - q - 0.5 * sigma * sigma) * t
    vol = sigma * math.sqrt(t)
    disc = math.exp(-r * t)
    for b in prange(n_blocks):
        start = b * BLOCK
        res = _european_block(
            s0, k, drift, vol, disc, is_call, start, min(n_paths, start + BLOCK), seed, b
        )
        out[b, 0], out[b, 1], out[b, 2] = res
    return out


def _asian_blocks(s0, k, t, r, sigma, q, is_call, n_steps, n_paths, seed):
    n_blocks = (n_paths + BLOCK - 1) // BLOCK
    out = np.zeros((n_blocks, 3))
    dt = t / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    disc = math.exp(-r * t)
    for b in prange(n_blocks):
        start = b * BLOCK
        res = _asian_block(
            s0, k, drift, vol, disc, is_call, n_steps, start, min(n_paths, start + BLOCK), seed, b
        )
        out[b, 0], out[b, 1], out[b, 2] = res
    return out


def _heston_blocks(s0, k, t, r, q, kappa, theta, xi, rho, v0, is_call, n_steps, n_paths, seed):
    n_blocks = (n_paths + BLOCK - 1) // BLOCK
    out = np.zeros((n_blocks, 3))
    dt, ek, k0, k1, k2, k3, k4 = _qe_constants(t, n_steps, kappa, theta, xi, rho)
    drift_dt = (r - q) * dt
    disc = math.exp(-r * t)
    for b in prange(n_blocks):
        start = b * BLOCK
        res = _heston_block(
            s0,
            k,
            disc,
            is_call,
            n_steps,
            v0,
            theta,
            xi,
            kappa,
            ek,
            k0,
            k1,
            k2,
            k3,
            k4,
            drift_dt,
            start,
            min(n_paths, start + BLOCK),
            seed,
            b,
        )
        out[b, 0], out[b, 1], out[b, 2] = res
    return out


def _heston_from_normals(s0, k, t, r, q, kappa, theta, xi, rho, v0, is_call, zv, zx):
    n_paths, n_steps = zv.shape
    dt, ek, k0, k1, k2, k3, k4 = _qe_constants(t, n_steps, kappa, theta, xi, rho)
    drift_dt = (r - q) * dt
    disc = math.exp(-r * t)
    out = np.empty(n_paths)
    for i in prange(n_paths):
        v = v0
        x = math.log(s0)
        for j in range(n_steps):
            v, x = _qe_step(
                v, x, zv[i, j], zx[i, j], theta, xi, kappa, ek, k0, k1, k2, k3, k4, drift_dt
            )
        st = math.exp(x)
        out[i] = disc * (max(st - k, 0.0) if is_call else max(k - st, 0.0))
    return out


def _asian_from_normals(s0, k, t, r, sigma, q, is_call, z):
    n_paths, n_steps = z.shape
    dt = t / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    vol = sigma * math.sqrt(dt)
    disc = math.exp(-r * t)
    out = np.empty(n_paths)
    for i in prange(n_paths):
        log_s = math.log(s0)
        total = 0.0
        for j in range(n_steps):
            log_s += drift + vol * z[i, j]
            total += math.exp(log_s)
        avg = total / n_steps
        out[i] = disc * (max(avg - k, 0.0) if is_call else max(k - avg, 0.0))
    return out


# Helpers and per-block functions: single variant each, safe to cache on disk.
_rotl = njit(cache=True, inline="always")(_rotl)
_splitmix = njit(cache=True, inline="always")(_splitmix)
_seed_state = njit(cache=True, inline="always")(_seed_state)
_uniform = njit(cache=True, inline="always")(_uniform)
_normal = njit(cache=True, inline="always")(_normal)
_qe_step = njit(cache=True, inline="always")(_qe_step)
_qe_constants = njit(cache=True)(_qe_constants)
_european_block = njit(cache=True)(_european_block)
_asian_block = njit(cache=True)(_asian_block)
_heston_block = njit(cache=True)(_heston_block)

# Each driver is compiled twice from the same source: serial and multi-threaded. Drivers are
# deliberately not cached: both variants come from one Python function, Numba's on-disk cache
# keys them identically, and the parallel dispatcher would silently load the serial machine
# code (this happened during development; parallel "matched" serial until the cache was off).
# Compiling them takes about a second per process.
KERNELS = {
    name: {"serial": njit(fn), "parallel": njit(parallel=True)(fn)}
    for name, fn in (
        ("european", _european_blocks),
        ("asian", _asian_blocks),
        ("heston", _heston_blocks),
        ("heston_from_normals", _heston_from_normals),
        ("asian_from_normals", _asian_from_normals),
    )
}


def price_european_numba(
    s0, k, t, r, sigma, q=0.0, kind="call", n_paths=1 << 20, seed: Seed = 0, parallel=True
) -> MCResult:
    fn = KERNELS["european"]["parallel" if parallel else "serial"]
    blocks = fn(s0, k, t, r, sigma, q, kind == "call", n_paths, _block_seed(seed))
    return _combine(blocks, n_paths, "numba_parallel" if parallel else "numba_serial")


def price_asian_numba(
    s0,
    k,
    t,
    r,
    sigma,
    q=0.0,
    kind="call",
    n_steps=252,
    n_paths=1 << 16,
    seed: Seed = 0,
    parallel=True,
) -> MCResult:
    fn = KERNELS["asian"]["parallel" if parallel else "serial"]
    blocks = fn(s0, k, t, r, sigma, q, kind == "call", n_steps, n_paths, _block_seed(seed))
    return _combine(blocks, n_paths, "numba_parallel" if parallel else "numba_serial")


def price_heston_numba(
    s0, k, t, r, p, q=0.0, kind="call", n_steps=252, n_paths=1 << 16, seed: Seed = 0, parallel=True
) -> MCResult:
    fn = KERNELS["heston"]["parallel" if parallel else "serial"]
    blocks = fn(
        s0,
        k,
        t,
        r,
        q,
        p.kappa,
        p.theta,
        p.xi,
        p.rho,
        p.v0,
        kind == "call",
        n_steps,
        n_paths,
        _block_seed(seed),
    )
    return _combine(blocks, n_paths, "numba_parallel" if parallel else "numba_serial")
