# twpa_design - Engineering Notes

Living document tracking design decisions, physical reasoning, and implementation notes for the `twpa_design` package.

---

## 2026-05-01: Tapered TWPA design — impedance and nonlinearity

### Context

A TWPA can taper two things at its edges to smooth the transition between the device and the environment:

- **Impedance taper** (`Z_taper`): the cell characteristic impedance ramps from `Z_env` (matched to the environment, default 50 Ω) at the device edges to `Z_TWPA` (the design value) in the center. Klopfenstein-optimal or linear shape. Suppresses port reflections.
- **Floquet nonlinearity taper** (`floquet_taper`): the per-cell nonlinear element strength ramps from weak at the edges to full strength in the center. Gaussian or Tukey shape. Suppresses gain ripple, improves pump impedance matching, and acts as an effective windowing of the nonlinear medium.

The two tapers are independent design choices. Each has its own enable flag, width (`Z_taper_width`, `floquet_taper_width`), and profile-shape parameters. They are mutually exclusive with the apodization windowing (`window_type`/`alpha`) of the periodic capacitance modulation.

`Z_taper` is auto-enabled when `Z0_TWPA_ohm != Z0_ohm`. Setting either taper to `True` triggers per-cell linear design recomputation along the line.

### Filters in the taper

Both tapers **include filter structures everywhere** — including in the taper region. The workflow at each cell is:

1. Compute all linear components (filter network from Foster transformations, shunt susceptance from periodic modulation) using the local cell parameters `Z(n)` and `fc(n)`.
2. If `floquet_taper=True`, take the series inductor from the filter design and split it into a nonlinear part + linear remainder, according to the nonlinearity weight profile `w(n)`.

The filter design produces a total series inductor value (`LinfLF1_H` for Foster form 1, `L0LF2_H` for form 2). The NL inductor replaces part of it, with a linear remainder making up the difference: `remainder = max(0, total − n_jj · L0_H_cell)`. Both Foster forms are handled uniformly by `add_inductance` in the netlist builder.

This differs from apodization windowing, which excludes filter structures from the windowed regions.

### Linear design along the line

The linear design fully determines the ABCD chain, S-parameters, and dispersion of the device. It is computed independently of the nonlinearity. At each cell, the filter and periodic-modulation components are denormalized using the local cell characteristic impedance `Z(n)` and cutoff frequency `fc(n)`:

```
L_series = g_L · Z(n) / (2π · fc(n))
C_shunt  = g_C / (Z(n) · 2π · fc(n))
```

What `Z(n)` and `fc(n)` look like along the line is set by two design choices: the **impedance taper** and the **`taper_cutoff` flag**.

### Impedance taper: `Z_taper`, `Z_profile`, `klopfenstein_A`

When `Z_taper=True` (auto-enabled when `Z0_TWPA_ohm ≠ Z0_ohm`), the cell impedance `Z(n)` transitions smoothly from `Z_env` at the device edges to `Z_TWPA` in the center, over a taper region of width `Z_taper_width · Ntot_cell / 2` on each side. The endpoints are forced exactly:

- `Z(0) = Z_env`
- `Z(Z_taper_cells − 1) = Z_TWPA`
- `Z(n) = Z_TWPA` in the center region (and mirrored in the right taper)

`Z_profile` chooses the shape inside the taper region.

**Linear profile** (`Z_profile='linear'`, default): linear in cell index.
```
Z(n) = Z_env + (Z_TWPA − Z_env) · n / (Z_taper_cells − 1)
```

**Klopfenstein profile** (`Z_profile='klopfenstein'`): equiripple-optimal impedance transformer (Klopfenstein, 1956). For the same impedance ratio and taper length, it minimizes the maximum in-band reflection:
```
ln(Z(n)/sqrt(Z_env · Z_TWPA)) = Γ_0 · φ(2n/(Z_taper_cells−1) − 1, A) / φ(1, A)
```
where `Γ_0 = ½·ln(Z_TWPA/Z_env)` is the DC reflection magnitude, `A` is the design parameter, and
```
φ(z, A) = ∫₀^z I₁(A·√(1−y²)) / (A·√(1−y²)) dy
```
is the Klopfenstein function (`I₁` is the modified Bessel function of order 1, via `scipy.special.iv(1, …)`; the integral is evaluated by `scipy.integrate.quad`). The factor `φ(z, A)/φ(1, A)` normalizes the profile so the endpoints sit exactly on `Z_env` and `Z_TWPA` — the canonical Klopfenstein has small (~few %) "ear" jumps at the endpoints to maintain perfect equiripple; we trade a tiny amount of optimality for clean port matching at the device boundaries.

The design parameter `A` sets the bandwidth/ripple trade-off:
```
in-band ripple amplitude Γ_m ≈ Γ_0 / cosh(A)         (canonical Klopfenstein)
cutoff frequency          f_cutoff ≈ A · v_p / (2π · L_taper)
A = cosh⁻¹(Γ_0 / Γ_m)
```
Larger `A` ⇒ sharper cutoff but larger in-band ripple. **Default: auto-compute** from a target max ripple `Γ_m = 0.05` (5% in-band reflection) via `A = cosh⁻¹(Γ_0 / Γ_m)`. Users can override with an explicit `klopfenstein_A`.

When `Z_taper=False` (or `Z0_TWPA_ohm = Z0_ohm`): `Z(n) = Z_TWPA` everywhere, `Z_profile` and `klopfenstein_A` are no-ops.

### `taper_cutoff`: how `fc(n)` responds to a varying cell

The `taper_cutoff` flag controls whether the cutoff frequency or the shunt capacitance absorbs the per-cell variation. From `L = g_L·Z/(2π·fc)` and `C = g_C/(Z·2π·fc)`:

- **`taper_cutoff=False`** (default): `fc(n) = fc_center` is constant; `C(n) = g_C / (Z(n) · 2π · fc_center)` adjusts. The total series inductance scales as `L_total(n) ∝ Z(n)`.
- **`taper_cutoff=True`**: `C(n) = C_center` is constant; `fc(n) = fc_center · Z_center / (Z(n) · w_eff(n))` adjusts, where `w_eff(n)` is the effective nonlinearity weight (see below). Then `L_total(n) ∝ Z(n)² · w_eff(n)`.

`taper_cutoff=False` keeps `fc` flat across the device but produces position-dependent shunt capacitance; `taper_cutoff=True` keeps `C` flat but lets `fc` drift. The choice is usually driven by fabrication tolerances on the smallest capacitor or by the maximum allowable cutoff frequency.

When neither taper is active (uniform line), both modes collapse to the same constant cell; the flag is vacuous. When only `floquet_taper` is active (no Z taper), the flag still matters for JJ/rf_squid because the per-cell L0 itself varies with `w(n)` (see "Effective weight" below), and the line has to either re-tune `fc` (mode `True`) or absorb the variation in a series remainder (mode `False`). For KI with uniform Z, the flag is also vacuous because `L0(n)` is decoupled from `w(n)` — see "Nonlinearity coupling" below.

`calculate_filter_components` is called per cell with the local `Z(n)` and `fc(n)`, producing correctly denormalized values for all components in both branches.

### Floquet nonlinearity taper: `floquet_taper`, `floquet_profile`, `floquet_taper_width`

When `floquet_taper=True`, the per-cell nonlinear element strength ramps from weak at the edges to full strength in the center, controlled by a profile `w(n) ∈ [0, 1]`:

- `floquet_profile='gaussian'` (default): `w(n) = 1 − exp(−n²/(2σ²))` from each edge, σ chosen so `w = 0.95` at the taper/center boundary
- `floquet_profile='tukey'`: half-cosine ramp from 0 to 1 on each side, flat 1.0 in the center

The taper region has width `floquet_taper_width · Ntot_cell / 2` on each side. The center region uses `w = 1` (full nonlinearity) and is symbolic in the netlist (one supercell ABCD raised to a power for the linear response).

Below the symbolic threshold (`w ≤ 1 − 1e-4`), per-cell numeric values are emitted in the netlist; above, the symbolic supercell is used.

`floquet_taper` is mutually exclusive with the apodization windowing of the periodic capacitance modulation — if `floquet_taper=True`, `window_type` must be `'boxcar'`.

### Nonlinear split per cell

At each cell, given the linear total inductance `L_total(n)` from the linear design and the Floquet weight `w(n)`:

```
L0_H_cell  = L0_H(w(n))            # depends on JJ structure type / nonlinearity — see below
rem_cell   = max(0, L_total(n) − n_jj · L0_H_cell)
```

The total series inductance always equals `L_total(n)`, preserving the linear response exactly. The Floquet weight only controls how much of that total is nonlinear vs linear remainder.

### Effective weight: bare JJ vs rf_squid vs KI

The Floquet weight `w(n)` controls the nonlinearity scale, but the mapping from `w(n)` to the effective per-cell `L0` depends on the nonlinearity type.

**Bare JJ:** `L0 = Lj = φ₀/Ic`. Scaling the JJ critical current by `1/w(n)` gives `L0(n) = LJ0 · w(n)`. The effective weight for component scaling is `w_eff(n) = L0(n)/L0_center = w(n)` — a direct identity. At the edges (w small), Lj is small (large Ic), the cell has a small series inductance.

**RF-SQUID:** `L0 = Lg / (1 + β_L · cos(φ_dc))`, where `Lg` stays constant and `β_L` is scaled by `w(n)`. At the edges (w→0): `β_L_eff → 0`, so `L0 → Lg` — the cell becomes a linear inductor of value `Lg`, which is **larger** than the center L0. The effective weight is:

```
w_eff(n) = L0(n) / L0_center = (1 + β_L · cos(φ_dc)) / (1 + w(n) · β_L · cos(φ_dc))
```

This is greater than 1 at the edges — the opposite of the bare JJ case. When `taper_cutoff=True`, the component scaling uses `w_eff(n)` rather than `w(n)`. Edge cells then have larger L and C, lower cutoff, and slower phase velocity (less transparent). This is a consequence of scaling only the JJ while keeping `Lg` constant: reducing `β_L` reduces the nonlinearity but increases the effective inductance.

We scale `β_L` down at the edges (rather than up) because:
- Small `β_L` gives a more linear inductor (the JJ participates less in the SQUID loop), matching the physical intent of the Floquet taper
- Large `β_L` (>1) would enter the hysteretic regime

The edge cells are never fully opaque because `w_eff` is bounded: at worst (w=0), `w_eff = 1 + β_L · cos(φ_dc)`, which for typical designs (β_L < 1) gives `w_eff < 2`.

For the bare-JJ limit (β_L → ∞, Lg → ∞ with Lg/β_L = LJ0 fixed): `w_eff → 1/w(n)`, confirming that the rf_squid and bare JJ cases have opposite scaling behavior.

**KI:** `L0` (linear kinetic inductance) and `Istar` (nonlinearity scale) are **physically independent knobs** — `L0` is set by wire width × length per cell, `Istar` is set by wire cross-section. The nonlinearity tapers via `Istar(n) = Istar_center / w(n)`, so the cubic NL coefficient `1/Istar²` scales as `w(n)²` — matching the JJ Floquet design in current-based units.

Crucially, `L0(n)` does **not** vary with the nonlinearity profile. It varies only with `Z(n)` from the impedance taper: `L0(n) = (L0_center / L_total_center) · L_total(n)`. So `w_eff(n) = 1` for KI in the cell-component scaling — there is no nonlinearity-driven cell variation. The KI Taylor coefficient `c2 = (φ₀/L0)² / Istar²` does still vary along the line whenever either knob varies, so the netlist emits per-cell numeric polynomials in the taper region.

### Nonlinearity coupling: KI vs JJ

JJ has **one knob** — `Lj` controls both the linear inductance and the nonlinearity, because `Ic = φ₀/Lj`. Tapering one tapers the other automatically, so a Floquet nonlinearity taper on JJ unavoidably perturbs the local impedance: the cell components have to compensate (via `taper_cutoff=True/False`).

KI has **two knobs** — `L0` and `Istar` are independent. The nonlinearity taper (`Istar(n) = Istar_center / w(n)`) does not change the local linear inductance. The impedance taper (`Z(n)`) drives `L0(n)` (and `C(n)`, `fc(n)`) directly. So for KI:

| Z taper? | `taper_cutoff` | `L0(n)` | `C(n)` | `fc(n)` |
|---|---|---|---|---|
| Yes (Z_env ≠ Z_TWPA) | True (C const) | ∝ Z(n)² | constant | ∝ 1/Z(n) |
| Yes | False (fc const) | ∝ Z(n) | ∝ 1/Z(n) | constant |
| No (Z_env = Z_TWPA) | either | constant | constant | constant |

When the impedance is uniform, the linear cell is uniform along the line regardless of `taper_cutoff` — the only Floquet effect is the nonlinearity gradient via `Istar(n)`. (This differs from JJ, where the nonlinearity taper itself forces `L0(n) ∝ w(n)`, so the cell components still vary at constant Z.)

### `taper_cutoff` is forced `True` for rf_squid

For bare JJ both `taper_cutoff=False` and `True` are physically meaningful: edges have *less* inductance than the center, so a positive series remainder `rem = L_total_center − n_jj·L0_edge > 0` can fill the gap and keep both `fc` and `Z` constant (`False`), or alternatively `fc` can be re-tuned per cell so that `L_total` scales with the JJ inductance (`True`).

For rf_squid this symmetry breaks. Edges have *more* inductance than the center (`L0(edge) = Lg`, `L0(center) = Lg/(1+β_L)`), so a passive series remainder cannot subtract inductance to bring the cell back down. The only way to keep the cell impedance equal to the environment at the edges is to **increase the shunt capacitance** to match — which by definition means lowering `fc`. There is no design point at which constant `Z`, constant `fc`, and fixed `Lg` are simultaneously satisfied for the rf_squid Floquet taper.

Because of this, when `nonlinearity='JJ'` and `jj_structure_type='rf_squid'`, the package **forces `taper_cutoff=True`** at validation time (with a warning if the user explicitly passed `False`). The user retains full control for bare JJ and KI.

### Constant rf_squid plasma frequency along the line

Even when the cell impedance is matched everywhere (`taper_cutoff=True`), the rf_squid Floquet design has a subtler issue: the **plasma frequency of the rf_squid varies along the line**.

The rf_squid plasma frequency is set by the parallel combination of `Lj_dyn(n) || Lg` resonating with `Cj(n)`:

```
f_plasma(n) = 1 / (2π · sqrt((Lj_dyn || Lg)(n) · Cj(n)))
            = 1 / (2π · sqrt(Lg/(1 + β_L·w(n)·cos(φ_dc)) · Cj(n)))
```

The default `Cj` scaling `Cj(n) = CJ0 · w(n)` keeps the bare-JJ plasma frequency `1/(2π·sqrt(Lj_dyn·Cj))` constant (this is the JJ self-resonance, useful for the Taylor expansion to remain valid). But the rf_squid sees `Lj_dyn || Lg` and `Cj`, whose product is NOT constant:

- At center (w=1): `(Lg/(1+β_L)) · CJ0` → e.g., 60 GHz for typical params
- At edge (w→0): `Lg · 0` → infinity, but in practice the edge plasma frequency is much higher (e.g., 200 GHz)

This 3× variation in plasma frequency along the line causes high pump harmonics (which can propagate through the taper) to encounter position-dependent dispersion. The harmonic balance solver struggles to converge: too few harmonics misses the dynamics, too many exposes numerical instabilities. Empirically this shows up as poor convergence for rf_squid Floquet TWPAs.

**`rf_squid_constant_plasma` config option (default True for rf_squid):** adds an extra shunt capacitance `Cjx(n)` in parallel with the rf_squid such that the *total* capacitance compensates the variation in `Lj_dyn || Lg`:

```
Cj_total(n) = Cj_center · (1 + β_L·w(n)·cos(φ_dc)) / (1 + β_L·cos(φ_dc))
Cjx(n)      = Cj_total(n) − Cj(n) · w(n)       # = (1 − w(n)) · Cj_center / (1+β_L·cos(φ_dc))
```

At center (w=1): `Cjx = 0`, no extra cap. At edge (w→0): `Cjx = Cj_center / (1+β_L·cos(φ_dc))`, a finite cap. The result: `(Lj_dyn || Lg)(n) · Cj_total(n)` is constant, so the rf_squid plasma frequency is uniform along the line.

Convergence improves dramatically with this enabled — the line looks uniform to high pump harmonics. The extra cap is a fabricated metal cap (separate from the JJ's intrinsic capacitance), placed in parallel with the rf_squid loop. It appears in the netlist as `Cjx{component_name}` only in taper cells (Cjx=0 in the center region).

The flag is a no-op for bare JJ (where `Lj · Cj = const` is already preserved by the existing scaling) and KI (no Lg branch, no parallel-resonance issue).

### Edge cases

**No series inductor from filter design.** When the filter transformation does not produce a finite series inductor (e.g., `LinfLF1_H = ∞` or `L0LF2_H = ∞`, which can happen with `select_one_form='C'`), the `taper_cutoff` logic must not attempt `fc(n) = g_L · Z(n) / (2π · ∞ · w_eff(n))`. In this case, skip the cutoff taper for the series branch — `fc` stays constant, remainder stays infinite, and the Floquet weight still applies to `L0_H` for the nonlinearity ramp. The line is effectively unfiltered in the series branch, so the cutoff taper has nothing to act on.

**Foster form 1 vs 2 in the ABCD.** The only practical difference is that form 1 includes the JJ parasitic capacitance resonance `n_jj·jωL0/(1−L0·CJ·ω²)` while form 2 treats the inductor as purely linear `1/(jωL0LF2)`. This is a minor effect — the JJ plasma frequency is typically 40+ GHz, well above the operating band. The phase matching calculation uses `k_radpercell` (from the ABCD dispersion), but the CJ correction is small. The filter resonator components are independent of `L0_H` in both forms.

**Bare-JJ `xi_perA²` helper formula.** The per-cell helper `compute_floquet_cell_parameters` exposes `xi_perA²` with units `1/A²` for all three branches (bare JJ, rf_squid, KI). The bare-JJ formula is `0.5/Ic²`, consistent with the more general scalar form `(3·c2² − c1·c3)/(2·c1⁴)/Ic²` used by the designer's SPM/XPM phase-mismatch calculation (which reduces to `0.5/Ic²` for bare JJ at zero bias).

---

## 2026-04-09: TWPA-TWPA center filter design

### Context

The TWPA-TWPA topology cascades two TWPAs through diplexers:
```
LP(in) ─┐                           ┌─LP (center) ─ 50Ω     50Ω ─ LP (center)─┐                           ┌─ LP(out)
         ├─ common ─ TWPA1 ─ common ┤                                         ├─ common ─ TWPA2 ─ common ─┤
HP(in) ─┘                           └─────────────── HP_center ───────────────┘                           └─ HP(out)
```

The signal path (LP arms) is terminated with 50 Ohm loads at the internal junction, while the pump/idler path (HP arms) passes through a center HP filter connecting the two inner diplexer common ports. This center filter is **doubly open-terminated**: neither end sees a resistive load, both see the reactive impedance of a diplexer common node.

### Design challenge

Standard filter design assumes either:
- **Doubly terminated**: resistive loads on both ends (g0 and g_{n+1} real)
- **Singly terminated**: resistive on one end, open on the other

The center filter needs neither — it's doubly open. In classical filter theory, this can be handled via **immittance inverter design**: rewrite the ladder as resonators coupled through K/J inverters, where termination impedances only appear in the external inverter values. For open terminations (Z → ∞), the external inverters become large (weak coupling), and the internal coupling values are exact regardless of termination type. However, translating K/J inverters back to physical LC components requires absorbing negative elements at both open ends, which is mathematically messier than the singly-terminated case.

### Current implementation: empirical symmetric extension

For Butterworth prototypes, a simpler approach works well and is straightforward to implement with the existing g-value machinery:

1. Compute singly-terminated g-values for a high-order filter (e.g., N=25)
2. Observe that g-values near the open end converge to a constant (~2.0 for Butterworth)
3. Take the last M converged values and mirror them symmetrically
4. The center filter has component values: `[g_M, ..., g_2, g_1, g_1, g_2, ..., g_M]`
5. Optionally extend the constant-valued middle section for longer filters

This works because high-order Butterworth g-values approach `g_k → 2` far from the source end, so the interior of the filter resembles a uniform transmission line. The mirrored structure ensures symmetry between the two TWPA stages.

**Why this is valid for Butterworth**: The converged g-values are the limiting case of the immittance inverter coupling coefficients for uniform coupling. The empirical approach and the systematic inverter method give the same result.

**Limitation for Chebyshev**: Chebyshev g-values oscillate rather than converge, so simple mirroring doesn't apply. The immittance inverter method would handle Chebyshev correctly. This is a future improvement.

### Future improvement

Implement the immittance inverter approach for the center filter, which would:
- Support Chebyshev and other approximations natively
- Give exact component values without requiring high-order reference filters
- Provide a more systematic design path

---

## 2026-04-09: Full N-port S-matrix and harmonic extraction

### Context

The julia_wrapper previously extracted only 4 S-parameters (S11, S12, S21, S22) between the user-specified signal and output ports, and plot labels were hardcoded. This was insufficient for multi-port devices like diplexed TWPAs.

### Changes

- **`TWPAResults` dataclass**: Replaced `S11/S12/S21/S22` fields with `S_fund` (3D array: `n_ports x n_ports x n_freqs`) and `S_harmonic` (4D array: `n_modes x n_ports x n_ports x n_freqs`). Backward-compatible `@property` aliases (`S11`, `S12`, `S21`, `S22`, `idler_response`, `backward_idler_response`) preserve existing code.
- **Extraction**: Single vectorized Julia call `abs2.(S(:, :, (0,), :, :))` replaces 6 individual calls. Returns all modes x all ports x all ports in one shot.
- **Save/Load**: New `.npz` format stores `S_fund`, `S_harmonic`, `port_count`, `port_numbers`. Legacy files with `S11/S12/S21/S22` keys are detected and reconstructed automatically.
- **Plot labels**: Dynamic — `|S_{42}|` instead of hardcoded `|S_{21}|`.
- **Access methods**: `results.s_param(j, k)` for fundamental, `results.s_harmonic(n, j, k)` for harmonics, both 1-based port numbers.

### Implementation choices

- **3D/4D numpy arrays** over nested lists — clean serialization with `npz`, easy slicing.
- **Port numbers stored explicitly** — handles non-contiguous ports.
- **Vectorized extraction** exploits JosephsonCircuits.jl KeyedArray support for `:` (Colon) on port dimensions.

---

## 2026-04-09: Peripheral filter builder module

### Context

For diplexed TWPAs and TWPA-TWPA cascades, peripheral filter circuits (diplexers, multiplexers) need to be designed and composed with TWPA netlists into multi-port devices.

### Changes

- **New module `filter_builder.py`** (~3400 lines): Contains g-value computation (Butterworth, Chebyshev Type I), Foster frequency transformations, filter/multiplexer design, netlist generation, standalone response analysis, and topology composition.
- **`compose_chain()`**: Stitches alternating peripheral and TWPA netlists by stripping ports/resistors, merging junction nodes, and adding fresh external ports with sequential numbering. Connection topology specified via `(out_role, in_role)` tuples.
- **`save_raw_netlist_to_file()`**: Extracted from `netlist_JC_builder.save_netlist_to_file()` to share the file-writing logic between TWPA netlist builder and filter builder.
- **Node naming**: String-prefixed nodes (`m1_lp_1`, `m1_hp_2`, `m1_c`) avoid collisions with TWPA integer nodes regardless of TWPA size.

### Implementation choices

- **Single file** rather than 3 separate modules — matches the package convention of one `.py` per concern.
- **Numeric component values** (not symbolic) — peripheral filters have ~50 components, no parametric reuse needed unlike TWPAs with 2000+ identical cells.
- **Singly terminated filters** for multiplexer arms — g-values computed with open-circuit load, branch order reversed when building from common node outward so g1 is at the 50 Ohm port end and g_n at the open junction end.
- **Composition ordering**: Input-side peripherals have their arms reversed in the netlist so components read port-to-junction; output-side peripherals read junction-to-port. The netlist reads naturally from device edge inward and back out.

---

## 2026-02-09: Harmonics spatial plotting - Node selection

### Context

We added a `plot_harmonics()` method to `TWPAResults` that plots the power (in dBm) of pump and signal/idler tones as a function of position along the TWPA. The data comes from JosephsonCircuits.jl's `sol.nonlinear.nodeflux` (pump) and `sol.linearized.nodeflux` (signal/idler).

### Problem: Which nodes are we plotting?

The nodeflux arrays from JosephsonCircuits.jl contain one entry per **unique non-ground node** in the circuit. For a TWPA with embedded filter structures (e.g., the KTWPA with 5004 cells), the total number of nodes is significantly larger than the number of cells, because each filter section introduces internal side-branch nodes.

**Example: KTWPA filter section** (from `4wm_ktwpa_5004cells_01.py`):

```
Main line: ... → node 8 → node 9 → ??? → node 10 → node 13 → ...
                              ↓                ↑
              NL0LF2_8:  9 → 11 → 10    (nonlinear inductor + remainder)
              LiLF2_1_8: 9 → 12 → 10    (parallel LC branch)
                              ↓
              C0CF1_8:  10 → ground      (shunt cap)
```

Nodes 11 and 12 are **internal to the filter**: they sit on parallel branches between nodes 9 and 10. They are not "along the line" in a meaningful spatial sense — they represent intermediate points within a lumped filter element that occupies a single physical location on the chip.

If we plot the nodeflux at ALL nodes (including 11, 12), the x-axis doesn't correspond to a physical position along the transmission line. The extra nodes create artificial "bumps" or oscillations in the spatial profile, since the power at the internal filter nodes may differ from the main-line nodes due to the filter's internal impedance structure.

### Problem: Node ordering

A secondary issue: JosephsonCircuits.jl with `sorting="name"` sorts node names alphabetically, giving "1", "10", "100", "2", ... instead of numerical order. We fixed this by extracting the node names from the Julia keyed array (`AxisKeys.axiskeys`) and computing a numerical sort index. There is also a fallback that extracts node names from the netlist components.

### Solution: Graph-based main-line identification

**Definition:** A node is a "main-line node" if it lies on **every** path from port 1 to port 2. Equivalently, removing that node from the circuit graph (ignoring ground) disconnects port 1 from port 2.

**Algorithm:**
1. Build an undirected graph from the netlist: each component `(name, node1, node2, value)` creates an edge between `node1` and `node2`, ignoring any edge to ground (`'0'`).
2. Identify port nodes (port 1 and port 2 from the config).
3. For each non-ground node `n`:
   - Temporarily remove `n` and all its edges from the graph.
   - Check (via BFS/DFS) whether port 1 and port 2 are still connected.
   - If disconnected → `n` is a **main-line node**.
   - If still connected → `n` is a **side-branch node** (internal to a filter or other parallel structure).

**Why this works for all TWPA topologies:**

- **Simple JTWPA** (no filters): Every node is a main-line node, because the circuit is a simple chain. Removing any node breaks the chain.
- **KTWPA with embedded filters**: The filter's internal nodes (e.g., 11, 12) have parallel paths around them. Removing node 11 still leaves the path 9→12→10. But removing node 9 or node 10 disconnects the circuit, correctly identifying them as main-line.
- **Any future topology**: The algorithm makes no assumptions about component naming or circuit structure — it's purely topological.

**Performance:** For V ≈ 6000 nodes and E ≈ 12000 edges, the algorithm runs V BFS/DFS passes, each O(V + E). Total: O(V × (V + E)) ≈ 72M operations, which completes in well under a second in Python.

> **Note on BFS/DFS:** BFS (Breadth-First Search) and DFS (Depth-First Search) are standard graph traversal algorithms that visit all nodes reachable from a starting node. BFS explores neighbors level by level (using a queue), while DFS goes as deep as possible before backtracking (using a stack or recursion). For our purpose they are interchangeable — we just need to answer "can I still reach port 2 from port 1 after removing node X?"

**Implementation plan:**
- During result extraction (`_extract_results()`), compute the main-line node indices from the loaded netlist.
- Store a `main_line_node_indices` array in `TWPAResults` (persisted through save/load).
- In `plot_harmonics()`, use these indices to select only main-line nodes from the nodeflux arrays.
- The x-axis then maps `len(main_line_nodes)` points to `[1, total_cells]`, giving a meaningful spatial coordinate.

**Status:** Implemented and verified.

**Results (KTWPA 5004 cells):** 6117 total nodes → 5005 main-line nodes identified. The 1112 excluded nodes are internal filter side-branch nodes. 5005 = 5004 cells + 1, which is exactly the expected node count for a chain of N cells (N+1 nodes at the boundaries).

**Implementation details:**
- `_find_port_node()` and `_find_main_line_nodes()` added as module-level functions in `julia_wrapper.py`
- `main_line_node_indices` field added to `TWPAResults`, persisted through save/load
- `plot_harmonics()` automatically filters to main-line nodes when available, with fallback to all nodes for backward compatibility

---

## 2026-02-05: Harmonics spatial plotting - Initial implementation

### Motivation

When designing a TWPA, it is useful to visualize how pump power and signal/idler power evolve **along the device** (from input to output port). This reveals:
- Whether the pump is depleting along the line (gain saturation).
- How signal amplification builds up spatially.
- How idler modes grow and at what rate.
- Whether there are unexpected features (reflections, standing waves, mode coupling).

JosephsonCircuits.jl's harmonic balance solver provides this spatial information through the `nodeflux` field, which contains the complex flux amplitude at each node for each harmonic/mode.

### Physics: Nodeflux to power conversion

The nodeflux $\hat{\Phi}_n$ at a node is a complex amplitude (in the frequency domain) representing the magnetic flux associated with a given harmonic. To convert to a measurable power:

1. **Flux to current:** $I = \hat{\Phi}_n \cdot \omega \cdot \varphi_0 / Z_0$

   where $\omega$ is the angular frequency of the harmonic, $\varphi_0 = \Phi_0 / 2\pi = 3.291 \times 10^{-16}$ V·s is the reduced flux quantum, and $Z_0 = 50\,\Omega$ is the characteristic impedance.

2. **Current to power:** $P = \frac{1}{2} |I|^2 Z_0$

3. **Power in dBm:** $P_\text{dBm} = 10 \log_{10}(P \times 1000)$

### Data structure

- **Pump nodeflux** (`sol.nonlinear.nodeflux`): shape `(num_pump_harmonics, num_nodes)`. Always available for nonlinear simulations. The first row is the fundamental pump, second row is the 2nd harmonic, etc.

- **Signal/idler nodeflux** (`sol.linearized.nodeflux`): 5D keyed array with shape `(output_mode, node, input_mode, input_port, freq)`. Only available when `returnnodeflux=true` is passed to `hbsolve`. For a given signal frequency, we fix `input_mode = signal mode (0,)`, `input_port`, and `freq_index`, then iterate over `output_mode` to get different signal/idler spatial profiles.

### Mode labeling (4-wave mixing)

For 4WM, modes are labeled by an integer $n$ such that the mode frequency is $f = f_s + n \cdot f_p$:
- $n = 0$: **signal** at $f_s$
- $n = -2$: **first idler** at $2f_p - f_s$ (conjugate idler)
- $n = 2$: mode at $2f_p + f_s$
- $n = \pm 4$: higher-order mixing products

### Implementation choices

- **`store_signal_nodeflux` config flag**: Because the signal nodeflux is a large 5D array (can reach several GB for thousands of cells × many frequencies), storage is opt-in. Pump nodeflux is always stored since it's a small 2D array.

- **Color scheme**: pump = purple, signal = blue (plotted last/on top), first idler = orange, higher harmonics use distinct non-faded colors (brown, darkblue, red, green, pink, yellow).

- **Save/load**: All nodeflux data and metadata are saved in the `.npz` file so plots can be regenerated from saved results without re-running the simulation.

---
