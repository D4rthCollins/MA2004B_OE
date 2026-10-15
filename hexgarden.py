#!/usr/bin/env python3
"""
hexgarden.py - prototipo de plantacion hexagonal con flora preexistente.

Que hace
  1. Malla hexagonal (espaciado a, rotacion theta, offset) con subredes por nivel:
       grandes -> subred de indice 7, medianas -> subred de indice 3 (o 4), pequenas -> todos los nodos.
  2. Flora preexistente aleatoria (5 tipos). Bloquea nodos. Las "invasoras" son removibles (con costo).
  3. Influencia entre plantas por recurso, cada uno con su kernel de soporte compacto:
       agua    : solape de radios radiculares, compite segun tamano relativo y profundidad de raiz
       luz     : sombra proyectada (centro de sombra desplazado al norte, depende de alturas)
       nodriza : facilita a juveniles (reduce estres hidrico), efecto que desaparece con la edad
       N       : islas de fertilidad de la fijadora (Prosopis)
  4. Aptitud por planta = Rw^0.6 * Rn^0.15 * Rl^0.25 (Hill por recurso, factor limitante ponderado),
     evaluada a varias edades (el signo de la interaccion con Prosopis cambia con la edad).
  5. Recocido simulado: especie por nodo (o vacio), con presupuesto duro, tope por especie,
     delta incremental local, y pasada de remocion de invasoras con costo.
  6. Busqueda externa sobre a (y orientacion/offset por barrido barato).

TODOS los parametros biologicos de abajo son PLACEHOLDERS razonables, no datos medidos.
"""
import argparse
import csv
import json
import math
import os
import time
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

R3 = math.sqrt(3.0)

# ----------------------------------------------------------------------------------
# Tablas de especies
# ----------------------------------------------------------------------------------
# nombre, altura adulta m, diametro copa m, factor radicular (radio raiz / radio copa),
# uso de agua relativo, profundidad raiz 0-1, sensibilidad a sequia, sensibilidad a sombra,
# nodriza, fijadora N, tau crecimiento (anios), costo/unidad, nivel (1 grande, 2 mediano, 3 pequeno)
PLANTED = [
    ("Agave lechuguilla",        0.45, 0.35, 1.5, 0.20, 0.0, 0.5, 1.0, 0.0, 0.0,  6, 26.0, 3),
    ("Agave striata",            0.75, 0.85, 1.5, 0.25, 0.0, 0.6, 1.0, 0.0, 0.0,  6, 26.0, 3),
    ("Agave scabra",             0.85, 1.75, 1.4, 0.30, 0.0, 0.6, 0.9, 0.0, 0.0,  8, 26.0, 2),
    ("Agave salmiana",           2.50, 3.50, 1.3, 0.50, 0.0, 0.7, 0.7, 0.0, 0.0, 10, 26.0, 2),
    ("Opuntia cantabrigiensis",  1.20, 4.00, 1.5, 0.40, 0.0, 0.6, 0.8, 0.0, 0.0,  7, 18.0, 2),
    ("Opuntia engelmannii",      2.80, 2.15, 1.5, 0.40, 0.0, 0.6, 0.8, 0.0, 0.0,  6, 21.0, 2),
    ("Opuntia robusta",          4.00, 4.00, 1.5, 0.40, 0.0, 0.6, 0.8, 0.0, 0.0,  8, 18.0, 2),
    ("Opuntia streptacantha",    3.50, 4.00, 1.5, 0.45, 0.0, 0.6, 0.8, 0.0, 0.0, 10, 18.0, 1),
    ("Prosopis laevigata",       8.50, 7.00, 2.5, 1.00, 1.0, 0.5, 0.5, 1.0, 1.0,  9, 26.5, 1),
    ("Yucca filifera",           9.00, 3.25, 1.2, 0.40, 0.3, 0.6, 0.7, 0.0, 0.0, 25, 26.0, 1),
]
N_PLANT = len(PLANTED)

# nombre, altura, diametro copa, factor radicular, uso agua, profundidad, nodriza, fijadora,
# radio de exclusion (m), densidad por ha, removible, costo de remocion
EXISTING = [
    ("buffel (invasora)",  0.60, 0.70, 1.8, 3.0, 0.0, 0.0, 0.0, 0.25, 250, True,   5.0),
    ("lenosa invasora",    4.00, 5.00, 2.0, 1.2, 0.6, 0.0, 0.0, 1.50,  12, True,  40.0),
    ("arbusto nativo",     1.50, 2.00, 2.0, 0.5, 0.3, 0.6, 0.0, 0.90,  60, False,  0.0),
    ("arbol nativo",       5.00, 5.00, 2.2, 0.9, 0.8, 0.8, 0.0, 2.00,   6, False,  0.0),
    ("hierba nativa",      0.30, 0.50, 1.5, 0.6, 0.0, 0.0, 0.0, 0.20, 400, False,  0.0),
]
N_EX = len(EXISTING)
N_ALL = N_PLANT + N_EX
C_CH = 4  # canales: 0 agua, 1 luz, 2 nodriza, 3 nitrogeno


@dataclass
class Config:
    L: float = 100.0              # lado del terreno cuadrado (m)
    aridity: float = 0.7          # estres hidrico basal (0.3 humedo ... 1.2 hiperarido)
    nutr: float = 0.5             # pobreza de nutrientes basal
    budget: float = 35000.0       # presupuesto de plantacion + remocion
    mu: float = 0.01              # valor por unidad monetaria (tipo de cambio costo/beneficio)
    fill_bonus: float = 0.25      # premio por nodo ocupado (integridad de la malla)
    max_share: float = 0.40       # tope de proporcion por especie
    ages: tuple = (1.0, 6.0, 20.0)
    age_w: tuple = (0.30, 0.30, 0.40)
    range_m: float = 14.0         # alcance maximo de interaccion (truncamiento)
    shadow_k: float = 0.6         # desplazamiento de sombra = k * altura (hemisferio norte)
    dead_thr: float = 0.25        # aptitud bajo la cual la planta "no cuenta"
    big_index: int = 7
    mid_index: int = 3
    max_shift_frac: float = 0.18  # deformacion local maxima como fraccion de a
    steps_per_node: int = 12
    seed: int = 1
    a_grid: tuple = (2.0, 2.5, 3.0, 3.5, 4.5)
    n_orient: int = 60            # candidatos del barrido de orientacion/offset
    margin: float = 0.5
    nurse_gain: float = 3.0       # ganancia del efecto nodriza (calibrado al hecho estilizado)
    juv_vuln: float = 3.0         # vulnerabilidad hidrica extra de juveniles
    nurse_head_start: float = 0.0 # anios de ventaja con que se planta Prosopis (siembra escalonada)


# ----------------------------------------------------------------------------------
# Modelo de influencia
# ----------------------------------------------------------------------------------
def kern(x):
    """Kernel de soporte compacto (tricubo suave): 1 en x=0, 0 en x>=1."""
    return np.where(x < 1.0, (1.0 - x * x) ** 2, 0.0)


class Model:
    def __init__(self, cfg):
        self.cfg = cfg
        T = len(cfg.ages)
        t = np.array(cfg.ages)[None, :]
        P, E = PLANTED, EXISTING
        H0 = np.array([s[1] for s in P] + [s[1] for s in E])
        CD = np.array([s[2] for s in P] + [s[2] for s in E])
        RF = np.array([s[3] for s in P] + [s[3] for s in E])
        WU = np.array([s[4] for s in P] + [s[4] for s in E])
        self.DEPTH = np.array([s[5] for s in P] + [s[5] for s in E])
        self.WSENS = np.array([s[6] for s in P] + [1.0] * N_EX)
        self.SSENS = np.array([s[7] for s in P] + [1.0] * N_EX)
        self.NURSE = np.array([s[8] for s in P] + [s[6] for s in E])
        self.FIX = np.array([s[9] for s in P] + [s[7] for s in E])
        tau = np.array([float(s[10]) for s in P] + [1.0] * N_EX)
        f0 = np.array([0.08] * N_PLANT + [1.0] * N_EX)
        self.COST = np.array([s[11] for s in P])
        self.LEVEL = np.array([s[12] for s in P])
        off = np.zeros(N_ALL)
        off[8] = cfg.nurse_head_start  # Prosopis plantada antes que el resto
        self.f = f0[:, None] + (1 - f0[:, None]) * (1 - np.exp(-(t + off[:, None]) / tau[:, None]))
        self.H = H0[:, None] * self.f
        self.CR = (CD[:, None] / 2.0) * self.f
        self.ROOT = RF[:, None] * self.CR + 0.05
        self.U = WU[:, None] * self.ROOT ** 2
        self.JJ = np.clip(1.0 - self.f / 0.5, 0.0, 1.0)
        self.DOV = 1.0 - 0.6 * np.abs(self.DEPTH[:, None] - self.DEPTH[None, :])
        self.AGEW = np.array(cfg.age_w)
        self.WGT = np.array([math.sqrt(s[2]) for s in P])  # placeholder de valor ecologico
        self.T = T

    def contrib(self, src, rec, vx, vy):
        """Aporte de la fuente src sobre el receptor rec. (vx,vy) = vector fuente->receptor.
        Devuelve (n, 4, T): agua, luz, nodriza, nitrogeno."""
        n = len(vx)
        src = np.asarray(src) * np.ones(n, dtype=int)
        rec = np.asarray(rec) * np.ones(n, dtype=int)
        d = np.hypot(vx, vy)[:, None]
        Rs = self.ROOT[src] + self.ROOT[rec]
        share = self.U[src] / (self.U[src] + self.U[rec] + 1e-9)
        w = share * kern(d / Rs) * self.DOV[src, rec][:, None]
        sy = vy[:, None] - self.cfg.shadow_k * self.H[src]
        ds = np.hypot(vx[:, None], sy)
        hs = np.maximum(0.0, (self.H[src] - self.H[rec]) / (self.H[src] + 1e-9))
        li = hs * kern(ds / (self.CR[src] + self.CR[rec]))
        nu = self.NURSE[src][:, None] * kern(d / (self.CR[src] * 1.6 + 1e-9))
        fx = self.FIX[src][:, None] * kern(d / (self.CR[src] + 1e-9))
        return np.stack([w, li, nu, fx], axis=1)

    def fitness(self, rec, acc):
        """rec (n,), acc (n,4,T) -> aptitud (n,T)."""
        cfg = self.cfg
        J = self.JJ[rec]
        Bn = np.minimum(cfg.nurse_gain * acc[:, 2], 1.0)
        S0 = cfg.aridity * self.WSENS[rec][:, None]
        Sw = S0 * (1.0 + cfg.juv_vuln * J * (1.0 - 0.9 * Bn)) + acc[:, 0] * (1.0 - 0.5 * J)
        Rw = 1.0 / (1.0 + Sw ** 2)
        Sl = self.SSENS[rec][:, None] * acc[:, 1]
        Rl = 1.0 / (1.0 + Sl ** 2)
        Sn = cfg.nutr * (1.0 - 0.6 * np.minimum(acc[:, 3], 1.0))
        Rn = 1.0 / (1.0 + Sn ** 2)
        return Rw ** 0.6 * Rn ** 0.15 * Rl ** 0.25

    def value(self, rec, F):
        g = F / (1.0 + np.exp(-(F - self.cfg.dead_thr) / 0.05))
        return self.WGT[rec] * (g @ self.AGEW)


# ----------------------------------------------------------------------------------
# Malla y flora existente
# ----------------------------------------------------------------------------------
def sub_mask(i, j, idx):
    if idx == 1:
        return np.ones_like(i, dtype=bool)
    if idx == 3:
        return (i - j) % 3 == 0
    if idx == 4:
        return (i % 2 == 0) & (j % 2 == 0)
    if idx == 7:
        return (i + 5 * j) % 7 == 0
    raise ValueError("indice de subred no soportado (usa 1, 3, 4 o 7)")


def make_lattice(cfg, a, theta_deg, tx, ty):
    N = int(math.ceil(1.5 * cfg.L / a)) + 3
    ii, jj = np.meshgrid(np.arange(-N, N + 1), np.arange(-N, N + 1))
    i, j = ii.ravel(), jj.ravel()
    x = a * (i + 0.5 * j)
    y = a * (R3 / 2.0) * j
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    xr = c * x - s * y + cfg.L / 2 + tx
    yr = s * x + c * y + cfg.L / 2 + ty
    m = (xr >= cfg.margin) & (xr <= cfg.L - cfg.margin) & (yr >= cfg.margin) & (yr <= cfg.L - cfg.margin)
    return np.c_[xr[m], yr[m]], i[m], j[m]


def gen_existing(cfg, rng, buf=15.0):
    area_ha = (cfg.L + 2 * buf) ** 2 / 1e4
    pos, typ = [], []
    for t, e in enumerate(EXISTING):
        k = rng.poisson(e[9] * area_ha)
        pos.append(rng.uniform(-buf, cfg.L + buf, size=(k, 2)))
        typ.append(np.full(k, N_PLANT + t))
    pos = np.vstack(pos)
    typ = np.concatenate(typ)
    inside = (pos[:, 0] >= 0) & (pos[:, 0] <= cfg.L) & (pos[:, 1] >= 0) & (pos[:, 1] <= cfg.L)
    removable = np.array([EXISTING[t - N_PLANT][10] for t in typ]) & inside
    rm_cost = np.array([EXISTING[t - N_PLANT][11] for t in typ])
    excl = np.array([EXISTING[t - N_PLANT][8] for t in typ])
    return dict(pos=pos, typ=typ, removable=removable, rm_cost=rm_cost, excl=excl)


def cheap_orientation(cfg, a, ex, rng):
    """Barrido barato de (theta, tx, ty): cuenta nodos bloqueados por flora existente."""
    trees = []
    for t in range(N_EX):
        m = ex["typ"] == N_PLANT + t
        trees.append((cKDTree(ex["pos"][m]) if m.any() else None, EXISTING[t][8], EXISTING[t][10]))
    scores, params = [], []
    for _ in range(cfg.n_orient):
        th, tx, ty = rng.uniform(0, 60), rng.uniform(0, a), rng.uniform(0, a)
        P, _, _ = make_lattice(cfg, a, th, tx, ty)
        hard = np.zeros(len(P), bool)
        soft = np.zeros(len(P), bool)
        for tr, r, rem in trees:
            if tr is None:
                continue
            d, _ = tr.query(P)
            (soft if rem else hard).__setitem__(slice(None), (soft if rem else hard) | (d < r))
        scores.append(hard.sum() + 0.3 * (soft & ~hard).sum())
        params.append((th, tx, ty))
    scores = np.array(scores)
    b = int(np.argmin(scores))
    return params[b], dict(best=float(scores[b]), mean=float(scores.mean()), worst=float(scores.max()))


# ----------------------------------------------------------------------------------
# Plan: estado de la plantacion + operaciones incrementales
# ----------------------------------------------------------------------------------
class Plan:
    def __init__(self, cfg, M, ex, a, theta, tx, ty, rng):
        self.cfg, self.M, self.ex, self.rng = cfg, M, ex, rng
        self.a, self.theta, self.tx, self.ty = a, theta, tx, ty
        T = M.T
        P, i, j = make_lattice(cfg, a, theta, tx, ty)
        self.EP = ex["pos"]
        self.EX = ex["excl"]
        self.etype = ex["typ"]
        self.etree = cKDTree(self.EP)
        maxEX = max(e[8] for e in EXISTING)

        def blockers_at(p):
            ids = np.array(self.etree.query_ball_point(p, maxEX), dtype=int)
            if len(ids) == 0:
                return []
            d = np.hypot(self.EP[ids, 0] - p[0], self.EP[ids, 1] - p[1])
            return list(ids[d < self.EX[ids]])

        # deformacion local acotada: mover nodos bloqueados lo minimo que libere el bloqueo
        shift_max = cfg.max_shift_frac * a
        self.n_deformed = 0
        self.n_blocked_before = 0
        P = P.copy()
        for k in range(len(P)):
            if blockers_at(P[k]):
                self.n_blocked_before += 1
                done = False
                for mag in (0.35, 0.7, 1.0):
                    for ang in np.linspace(0, 2 * math.pi, 12, endpoint=False) + rng.uniform(0, 0.5):
                        q = P[k] + mag * shift_max * np.array([math.cos(ang), math.sin(ang)])
                        if not (cfg.margin <= q[0] <= cfg.L - cfg.margin and cfg.margin <= q[1] <= cfg.L - cfg.margin):
                            continue
                        if not blockers_at(q):
                            P[k] = q
                            self.n_deformed += 1
                            done = True
                            break
                    if done:
                        break
        self.P, self.i, self.j = P, i, j
        n = len(P)
        self.n = n
        self.tree = cKDTree(P)

        # bloqueos
        self.blk = np.zeros(n, int)
        self.e_blocks = [[] for _ in range(len(self.EP))]
        for k in range(n):
            for e in blockers_at(P[k]):
                self.blk[k] += 1
                self.e_blocks[e].append(k)

        # niveles (subredes)
        m_big = sub_mask(i, j, cfg.big_index)
        m_mid = sub_mask(i, j, cfg.mid_index) | m_big
        self.code = m_mid.astype(int) + 2 * m_big.astype(int)
        lv = M.LEVEL
        small = np.where(lv == 3)[0]
        mid = np.where(lv == 2)[0]
        big = np.where(lv == 1)[0]
        self.OPTS = {
            0: small,
            1: np.concatenate([small, mid]),
            3: np.concatenate([small, mid, big]),
        }
        self.OPTS[2] = self.OPTS[3]  # big sin mid no ocurre porque m_mid incluye m_big
        self.code = np.where(self.code >= 2, 3, self.code)

        # vecinos (CSR)
        pairs = self.tree.query_pairs(cfg.range_m, output_type="ndarray")
        ii_ = np.concatenate([pairs[:, 0], pairs[:, 1]])
        jj_ = np.concatenate([pairs[:, 1], pairs[:, 0]])
        o = np.argsort(ii_, kind="stable")
        ii_, jj_ = ii_[o], jj_[o]
        ptr = np.searchsorted(ii_, np.arange(n + 1))
        rel = P[jj_] - P[ii_]
        self.nb = [jj_[ptr[k]:ptr[k + 1]] for k in range(n)]
        self.rx = [rel[ptr[k]:ptr[k + 1], 0] for k in range(n)]
        self.ry = [rel[ptr[k]:ptr[k + 1], 1] for k in range(n)]

        # campo precalculado de la flora existente: E[nodo, especie receptora, canal, edad]
        self.E = np.zeros((n, N_PLANT, C_CH, T))
        for e in range(len(self.EP)):
            idx = np.array(self.tree.query_ball_point(self.EP[e], cfg.range_m), dtype=int)
            if len(idx):
                self.E[idx] += self.e_table(e, idx)

        # estado
        self.sp = -np.ones(n, int)
        self.acc = np.zeros((n, C_CH, T))
        self.val = np.zeros(n)
        self.removed = np.zeros(len(self.EP), bool)
        self.removable_ids = np.where(ex["removable"])[0]
        self.cost = 0.0
        self.counts = np.zeros(N_PLANT, int)
        self.npl = 0
        self.rm_total = 0

    # ---- utilidades
    def e_table(self, e, idx):
        n, R = len(idx), N_PLANT
        nodes = np.repeat(idx, R)
        rec = np.tile(np.arange(R), n)
        vx = self.P[nodes, 0] - self.EP[e, 0]
        vy = self.P[nodes, 1] - self.EP[e, 1]
        C = self.M.contrib(np.full(n * R, self.etype[e]), rec, vx, vy)
        return C.reshape(n, R, C_CH, self.M.T)

    def set_sp(self, k, new):
        M, sp = self.M, self.sp
        old = sp[k]
        nb, rx, ry = self.nb[k], self.rx[k], self.ry[k]
        pm = sp[nb] >= 0
        pl, plx, ply = nb[pm], rx[pm], ry[pm]
        sps = sp[pl]
        dV = 0.0
        if len(pl):
            if old >= 0:
                self.acc[pl] -= M.contrib(np.full(len(pl), old), sps, plx, ply)
            if new >= 0:
                self.acc[pl] += M.contrib(np.full(len(pl), new), sps, plx, ply)
            nv = M.value(sps, M.fitness(sps, self.acc[pl]))
            dV += nv.sum() - self.val[pl].sum()
            self.val[pl] = nv
        dV -= self.val[k]
        if new >= 0:
            a_ = self.E[k, new].copy()
            if len(pl):
                a_ += M.contrib(sps, np.full(len(pl), new), -plx, -ply).sum(0)
            self.acc[k] = a_
            nn = np.array([new])
            v = M.value(nn, M.fitness(nn, a_[None]))[0]
            self.val[k] = v
            dV += v
        else:
            self.acc[k] = 0.0
            self.val[k] = 0.0
        sp[k] = new
        return dV

    def apply_remove(self, e, remove):
        M = self.M
        idx = np.array(self.tree.query_ball_point(self.EP[e], self.cfg.range_m), dtype=int)
        sign = -1.0 if remove else 1.0
        dV = 0.0
        if len(idx):
            C = self.e_table(e, idx)
            self.E[idx] += sign * C
            pm = self.sp[idx] >= 0
            if pm.any():
                nodes = idx[pm]
                sps = self.sp[nodes]
                self.acc[nodes] += sign * C[pm][np.arange(pm.sum()), sps]
                nv = M.value(sps, M.fitness(sps, self.acc[nodes]))
                dV = nv.sum() - self.val[nodes].sum()
                self.val[nodes] = nv
        for k in self.e_blocks[e]:
            self.blk[k] += -1 if remove else 1
        return dV

    def share_ok(self, cur, new):
        if new < 0 or cur == new:
            return True
        tot = self.npl + (1 if cur < 0 else 0)
        if tot < 60:
            return True
        return self.counts[new] + 1 <= self.cfg.max_share * tot

    def _book(self, cur, new):
        if cur >= 0:
            self.counts[cur] -= 1
            self.npl -= 1
        if new >= 0:
            self.counts[new] += 1
            self.npl += 1

    # ---- optimizacion
    def init_random(self):
        cfg, rng = self.cfg, self.rng
        for k in rng.permutation(self.n):
            if self.blk[k] > 0:
                continue
            opts = self.OPTS[self.code[k]]
            s = opts[rng.integers(len(opts))]
            c = self.M.COST[s]
            if self.cost + c > cfg.budget or not self.share_ok(-1, s):
                continue
            self.set_sp(k, s)
            self._book(-1, s)
            self.cost += c

    def anneal(self, steps, T0, T1):
        cfg, rng, M = self.cfg, self.rng, self.M
        fb, mu = cfg.fill_bonus, cfg.mu
        acc_n = 0
        for step in range(steps):
            T = T0 * (T1 / T0) ** (step / max(1, steps))
            k = rng.integers(self.n)
            if self.blk[k] > 0:
                continue
            cur = self.sp[k]
            opts = self.OPTS[self.code[k]]
            new = -1 if rng.random() < 0.12 else opts[rng.integers(len(opts))]
            if new == cur:
                continue
            dcost = (M.COST[new] if new >= 0 else 0.0) - (M.COST[cur] if cur >= 0 else 0.0)
            if self.cost + dcost > cfg.budget + 1e-9 or not self.share_ok(cur, new):
                continue
            dV = self.set_sp(k, new)
            dJ = dV - mu * dcost + fb * (int(new >= 0) - int(cur >= 0))
            if dJ >= 0 or rng.random() < math.exp(dJ / T):
                self._book(cur, new)
                self.cost += dcost
                acc_n += 1
            else:
                self.set_sp(k, cur)
        return acc_n

    def removal_pass(self):
        cfg, rng, M = self.cfg, self.rng, self.M
        mu, fb = cfg.mu, cfg.fill_bonus
        cand = [e for e in self.removable_ids if not self.removed[e]]
        rng.shuffle(cand)
        n_done = 0
        for e in cand:
            rc = self.ex["rm_cost"][e]
            if self.cost + rc > cfg.budget:
                continue
            dJ = self.apply_remove(e, True) - mu * rc
            dcost = rc
            added = []
            freed = [k for k in self.e_blocks[e] if self.blk[k] == 0 and self.sp[k] < 0]
            for k in freed:
                best = None
                for s in self.OPTS[self.code[k]]:
                    c = M.COST[s]
                    if self.cost + dcost + c > cfg.budget:
                        continue
                    dv = self.set_sp(k, s)
                    dj = dv - mu * c + fb
                    self.set_sp(k, -1)
                    if dj > 0 and (best is None or dj > best[0]):
                        best = (dj, s)
                if best:
                    self.set_sp(k, best[1])
                    dJ += best[0]
                    dcost += M.COST[best[1]]
                    added.append((k, best[1]))
            if dJ > 0:
                self.removed[e] = True
                self.cost += dcost
                self.rm_total += rc
                for k, s in added:
                    self._book(-1, s)
                n_done += 1
            else:
                for k, s in reversed(added):
                    self.set_sp(k, -1)
                self.apply_remove(e, False)
        return n_done

    def optimize(self, verbose=True):
        cfg = self.cfg
        total = cfg.steps_per_node * self.n
        if self.npl == 0:
            self.init_random()
        self.V_init = float(self.val.sum())
        t0 = time.time()
        T0, T1 = 0.35, 0.01
        parts = [0.40, 0.35, 0.25]
        step_T = [(T0, T0 * (T1 / T0) ** 0.40), (T0 * (T1 / T0) ** 0.40, T0 * (T1 / T0) ** 0.75), (T0 * (T1 / T0) ** 0.75, T1)]
        removed_now = 0
        for q, (frac, (ta, tb)) in enumerate(zip(parts, step_T)):
            self.anneal(int(total * frac), ta, tb)
            if q < 2:
                removed_now += self.removal_pass()
        self.anneal(int(total * 0.05), T1, T1)
        self.time = time.time() - t0
        if verbose:
            print(f"   a={self.a:.2f}  nodos={self.n}  plantas={self.npl}  gasto={self.cost:.0f}  "
                  f"V={self.val.sum():.1f}  removidas={int(self.removed.sum())}  ({self.time:.0f}s)", flush=True)

    # ---- verificacion y resumen
    def full_check(self):
        M, sp = self.M, self.sp
        V, maxerr = 0.0, 0.0
        for k in np.where(sp >= 0)[0]:
            nb = self.nb[k]
            m = sp[nb] >= 0
            a_ = self.E[k, sp[k]].copy()
            if m.any():
                a_ += M.contrib(sp[nb][m], np.full(m.sum(), sp[k]), -self.rx[k][m], -self.ry[k][m]).sum(0)
            maxerr = max(maxerr, float(np.abs(a_ - self.acc[k]).max()))
            nn = np.array([sp[k]])
            V += M.value(nn, M.fitness(nn, a_[None]))[0]
        Et = np.zeros_like(self.E)
        for e in range(len(self.EP)):
            if self.removed[e]:
                continue
            idx = np.array(self.tree.query_ball_point(self.EP[e], self.cfg.range_m), dtype=int)
            if len(idx):
                Et[idx] += self.e_table(e, idx)
        return dict(V_full=V, dV=abs(V - self.val.sum()), acc_err=maxerr, E_err=float(np.abs(Et - self.E).max()))

    def J(self):
        cfg = self.cfg
        return float(self.val.sum() - cfg.mu * self.cost + cfg.fill_bonus * self.npl)

    def summary(self):
        free = int((self.blk == 0).sum())
        return dict(a=self.a, theta=self.theta, nodes=self.n, free_nodes=free, planted=int(self.npl),
                    empty_free=free - int(self.npl), blocked=int((self.blk > 0).sum()),
                    deformed=self.n_deformed, spend=float(self.cost), V=float(self.val.sum()), J=self.J(),
                    removed=int(self.removed.sum()),
                    counts={PLANTED[s][0]: int(self.counts[s]) for s in range(N_PLANT)})

    def fitness_table(self):
        pl = np.where(self.sp >= 0)[0]
        F = self.M.fitness(self.sp[pl], self.acc[pl])
        return pl, F


# ----------------------------------------------------------------------------------
# Salidas
# ----------------------------------------------------------------------------------
def save_outputs(cfg, plan, results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Circle

    os.makedirs(outdir, exist_ok=True)
    pl, F = plan.fitness_table()
    with open(os.path.join(outdir, "plan_nodes.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "especie", "nivel"] + [f"aptitud_{int(t)}a" for t in cfg.ages])
        for r, k in enumerate(pl):
            s = plan.sp[k]
            w.writerow([round(plan.P[k, 0], 3), round(plan.P[k, 1], 3), PLANTED[s][0], PLANTED[s][12]] +
                       [round(float(v), 3) for v in F[r]])
    with open(os.path.join(outdir, "flora_existente.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["x", "y", "tipo", "removida"])
        for e in range(len(plan.EP)):
            if 0 <= plan.EP[e, 0] <= cfg.L and 0 <= plan.EP[e, 1] <= cfg.L:
                w.writerow([round(plan.EP[e, 0], 3), round(plan.EP[e, 1], 3),
                            EXISTING[plan.etype[e] - N_PLANT][0], int(plan.removed[e])])
    with open(os.path.join(outdir, "summary.json"), "w") as fh:
        json.dump(dict(config=cfg.__dict__, best=plan.summary(), sweep=results), fh, indent=1, default=str)

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(15, 7.2), gridspec_kw=dict(width_ratios=[1.6, 1]))
    cmap = plt.get_cmap("tab10")
    # flora existente
    ecol = {0: "#c9a227", 1: "#b03a2e", 2: "#7d8c5a", 3: "#2f5d3a", 4: "#cfcfcf"}
    for e in range(len(plan.EP)):
        x, y = plan.EP[e]
        if not (0 <= x <= cfg.L and 0 <= y <= cfg.L):
            continue
        t = plan.etype[e] - N_PLANT
        if plan.removed[e]:
            ax.plot(x, y, "x", color="red", ms=7, mew=1.6, zorder=6)
        else:
            ax.plot(x, y, ".", color=ecol[t], ms=3 if t in (0, 4) else 7, alpha=0.9, zorder=2)
    # plantadas (circulo = copa adulta)
    for s in range(N_PLANT):
        ids = pl[plan.sp[pl] == s]
        if len(ids) == 0:
            continue
        r = PLANTED[s][2] / 2.0
        pc = PatchCollection([Circle(tuple(plan.P[k]), r) for k in ids], facecolor=cmap(s), alpha=0.30,
                             edgecolor=cmap(s), linewidth=0.5, zorder=3)
        ax.add_collection(pc)
        ax.plot(plan.P[ids, 0], plan.P[ids, 1], ".", color=cmap(s), ms=3, zorder=4,
                label=f"{PLANTED[s][0]} ({len(ids)})")
    ax.set_xlim(0, cfg.L)
    ax.set_ylim(0, cfg.L)
    ax.set_aspect("equal")
    ax.set_title(f"Plan: a={plan.a:.2f} m, theta={plan.theta:.1f} deg, {plan.npl} plantas, "
                 f"gasto {plan.cost:.0f}, {int(plan.removed.sum())} remociones (x rojas)\n"
                 "circulos = copa adulta; puntos grises/dorados/rojos = flora existente", fontsize=10)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.06), ncol=2, fontsize=8, frameon=False)
    # barrido de a
    A = [r["a"] for r in results]
    ax2.bar([str(a) for a in A], [r["J"] for r in results], color="#4a7c59")
    for x_, r in enumerate(results):
        ax2.text(x_, r["J"], f"{r['planted']} pl\n{r['empty_free']} vacios", ha="center", va="bottom", fontsize=8)
    ax2.set_xlabel("espaciado base a (m)")
    ax2.set_ylabel("objetivo J = V - mu*gasto + premio de ocupacion")
    ax2.set_title("Barrido de a (presupuesto duro)", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "plan_map.png"), dpi=130)
    plt.close(fig)


def run(cfg, outdir):
    rng = np.random.default_rng(cfg.seed)
    M = Model(cfg)
    ex = gen_existing(cfg, rng)
    inside = np.sum(ex["removable"])
    print(f"Terreno {cfg.L:.0f}x{cfg.L:.0f} m, flora existente en terreno: "
          f"{int(((ex['pos'] >= 0) & (ex['pos'] <= cfg.L)).all(1).sum())}, removibles: {int(inside)}", flush=True)
    results, best = [], None
    for a in cfg.a_grid:
        (th, tx, ty), st = cheap_orientation(cfg, a, ex, rng)
        gain = 100 * (st["mean"] - st["best"]) / max(st["mean"], 1e-9)
        print(f"a={a}: barrido orientacion -> mejor {st['best']:.0f} vs media {st['mean']:.0f} "
              f"(ganancia {gain:.1f}%)", flush=True)
        plan = Plan(cfg, M, ex, a, th, tx, ty, rng)
        plan.optimize()
        r = plan.summary()
        r["orient_gain_pct"] = gain
        results.append(r)
        if best is None or r["J"] > best.J():
            best = plan
    chk = best.full_check()
    print("Verificacion incremental vs recalculo completo:", chk)
    save_outputs(cfg, best, results, outdir)
    print(f"{'a':>5} {'nodos':>6} {'libres':>6} {'plant.':>6} {'vacios':>6} {'gasto':>7} {'V':>8} {'J':>8} {'rem.':>4}")
    for r in results:
        print(f"{r['a']:5.2f} {r['nodes']:6d} {r['free_nodes']:6d} {r['planted']:6d} {r['empty_free']:6d} "
              f"{r['spend']:7.0f} {r['V']:8.1f} {r['J']:8.1f} {r['removed']:4d}")
    print("Mejor a:", best.a, "| composicion:", best.summary()["counts"])
    return best, results


# ----------------------------------------------------------------------------------
# Autotest
# ----------------------------------------------------------------------------------
def selftest():
    ok = True
    cfg = Config(L=50.0, steps_per_node=8, budget=9000.0, seed=3)
    M = Model(cfg)
    rng = np.random.default_rng(cfg.seed)
    ex = gen_existing(cfg, rng)
    plan = Plan(cfg, M, ex, 3.0, 17.0, 0.7, 1.1, rng)
    plan.init_random()
    v0 = plan.val.sum()
    plan.optimize(verbose=False)
    chk = plan.full_check()
    print("T1 delta incremental vs recalculo completo:", chk, "| removidas:", int(plan.removed.sum()))
    t1 = chk["dV"] < 1e-6 and chk["acc_err"] < 1e-8 and chk["E_err"] < 1e-8
    print("   ->", "OK" if t1 else "FALLA")
    ok &= t1

    # T2: efecto nodriza dependiente de la edad (receptor O. robusta idx 6 y Yucca idx 9)
    ok2 = True
    for rec_i in (6, 9):
        rec = np.array([rec_i])
        alone = np.zeros((1, C_CH, M.T))
        Fa = M.fitness(rec, alone)[0]
        for nm, src, d in (("arbusto nativo a 0.8 m", N_PLANT + 2, 0.8), ("arbol nativo a 2 m", N_PLANT + 3, 2.0)):
            near = M.contrib(np.array([src]), rec, np.array([d]), np.array([0.0]))
            Fn = M.fitness(rec, near)[0]
            good = Fn[0] > Fa[0] and Fn[-1] < Fa[-1]
            ok2 &= good
            print(f"T2 {PLANTED[rec_i][0]:<20} {nm:<22} sola={np.round(Fa, 3)} con={np.round(Fn, 3)} "
                  f"-> {'OK' if good else 'FALLA'}")
    # Prosopis plantada al mismo tiempo no ayuda al juvenil; con ventaja de 4 anios si
    cfg2 = Config(nurse_head_start=4.0)
    M2 = Model(cfg2)
    rec = np.array([6])
    near0 = M.contrib(np.array([8]), rec, np.array([1.5]), np.array([0.0]))
    near4 = M2.contrib(np.array([8]), rec, np.array([1.5]), np.array([0.0]))
    a0 = M.fitness(rec, near0)[0][0]
    a4 = M2.fitness(rec, near4)[0][0]
    alone1 = M.fitness(rec, np.zeros((1, C_CH, M.T)))[0][0]
    print(f"   Prosopis misma fecha: aptitud juvenil {a0:.3f} | con 4 anios de ventaja: {a4:.3f} | sola: {alone1:.3f}")
    t2 = ok2 and a4 > a0
    print("   T2 ->", "OK" if t2 else "FALLA")
    ok &= t2

    # T3: invariantes
    sp = plan.sp
    planted = sp >= 0
    t3a = not np.any(planted & (plan.blk > 0))
    t3b = plan.cost <= cfg.budget + 1e-6
    big = np.isin(sp, np.where(M.LEVEL == 1)[0])
    m7 = sub_mask(plan.i, plan.j, cfg.big_index)
    t3c = not np.any(big & ~m7)
    mid = np.isin(sp, np.where(M.LEVEL == 2)[0])
    m3 = sub_mask(plan.i, plan.j, cfg.mid_index) | m7
    t3d = not np.any(mid & ~m3)
    cost_check = float(sum(M.COST[s] * (sp == s).sum() for s in range(N_PLANT))) + plan.rm_total
    t3e = abs(cost_check - plan.cost) < 1e-6
    print(f"T3 invariantes: sin plantas en nodos bloqueados={t3a}, presupuesto={t3b}, "
          f"grandes solo en subred 7={t3c}, medianas solo en subred 3/7={t3d}, contabilidad de gasto={t3e}")
    ok &= all([t3a, t3b, t3c, t3d, t3e])

    # T4: el optimizador mejora a la asignacion aleatoria inicial
    t4 = plan.val.sum() > v0
    print(f"T4 V aleatorio inicial={v0:.1f} -> optimizado={plan.val.sum():.1f}  ->", "OK" if t4 else "FALLA")
    ok &= t4
    print("AUTOTEST", "OK" if ok else "FALLA")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--L", type=float, default=100.0)
    ap.add_argument("--budget", type=float, default=35000.0)
    ap.add_argument("--aridity", type=float, default=0.7)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--a-grid", type=float, nargs="+", default=[2.0, 2.5, 3.0, 3.5, 4.5])
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    if args.selftest:
        raise SystemExit(0 if selftest() else 1)
    cfg = Config(L=args.L, budget=args.budget, aridity=args.aridity, steps_per_node=args.steps,
                 seed=args.seed, a_grid=tuple(args.a_grid))
    run(cfg, args.out)
