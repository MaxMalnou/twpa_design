# Filter Builder Module - Complete Reference

This module provides tools for designing peripheral filter circuits (standalone filters, diplexers, and N-way multiplexers) and integrating them into TWPA netlists for multi-port device simulation with JosephsonCircuits.jl.

## Quick Start

```python
from twpa_design.filter_builder import (
    FilterSpec, design_filter, design_multiplexer,
    filter_to_netlist, multiplexer_to_netlist,
    compose_chain, save_peripheral_netlist,
    plot_peripheral_response
)

# Design a diplexer
mux = design_multiplexer([
    FilterSpec('lp', order=15, fc=8e9),
    FilterSpec('hp', order=15, fc=8e9),
])

# Plot standalone response
plot_peripheral_response(mux)

# Generate netlist and compose with TWPA
mux_net = multiplexer_to_netlist(mux, prefix='m1')
comps, params, meta = compose_chain([mux_net, twpa_dict, mux_net2])
save_peripheral_netlist(comps, params, meta, "composed_netlist.py")
```

## Classes

### FilterSpec

Specification for a single filter arm.

```python
FilterSpec(
    response='lp',             # str: 'lp', 'hp', 'bp', or 'bs'
    order=15,                  # int: Filter order
    fc=8e9,                    # float: Cutoff/center frequency in Hz
    approx='butterworth',      # str: 'butterworth' or 'chebyshev1'
    ripple_dB=None,            # float: Passband ripple (required for chebyshev1)
    Z0=50.0,                   # float: Reference impedance in Ohms
    bw=None,                   # float: Bandwidth in Hz (required for 'bp' and 'bs')
    termination='double',      # str: 'double' (standalone) or 'single' (multiplexer arm)
    foster_form=1,             # int: Foster canonical form (1 or 2)
    label=None,                # str: Label for the arm (auto-generated if None)
)
```

### FilterDesign

Result of `design_filter()`. Contains:
- `spec`: The original FilterSpec
- `g_values`: LP prototype g-values `[g0, g1, ..., gn, g_{n+1}]`
- `transfo_result`: Dict with `'series'` and `'shunt'` branch dicts containing L/C values in SI units (H, F)

### MultiplexerDesign

Result of `design_multiplexer()`. Contains:
- `arms`: List of FilterDesign (one per arm, all singly terminated)
- `Z0`: Common reference impedance
- `label`: Descriptive label

### PeripheralNetlist

A peripheral circuit in JC netlist format. Contains:
- `components`: List of `(name, node1, node2, value)` tuples
- `parameters`: Dict of symbolic parameter name -> value (e.g., `{'R_port': 50}`)
- `port_map`: Dict mapping port roles to `(port_number, node_string)`. Examples:
  - Single filter: `{'input': (1, 'f1_in'), 'output': (2, 'f1_out')}`
  - Diplexer: `{'common': (1, 'm1_c'), 'lp': (2, 'm1_lp_out'), 'hp': (3, 'm1_hp_out')}`
- `metadata`: Dict with design type, component count, arm labels, etc.
- `n_ports`: Number of ports (property)
- `port_numbers`: Sorted list of port numbers (property)

## Design Functions

### design_filter

Design a standalone doubly-terminated filter.

```python
design = design_filter(FilterSpec('lp', order=5, fc=8e9))
design = design_filter(FilterSpec('hp', order=7, fc=8e9, approx='chebyshev1', ripple_dB=0.5))
design = design_filter(FilterSpec('bp', order=5, fc=8e9, bw=2e9))
```

### design_multiplexer

Design an N-way multiplexer. Each arm is forced to single termination. Arms can be any mix of filter types.

```python
# LP/HP diplexer
mux = design_multiplexer([
    FilterSpec('lp', 15, 8e9),
    FilterSpec('hp', 15, 8e9),
])

# Triplexer: LP + BP + HP
mux = design_multiplexer([
    FilterSpec('lp', 5, 6e9),
    FilterSpec('bp', 5, 8e9, bw=2e9, label='bp1'),
    FilterSpec('hp', 5, 10e9),
])

# Two bandpass arms
mux = design_multiplexer([
    FilterSpec('bp', 7, 6e9, bw=1e9, label='bp1'),
    FilterSpec('bp', 7, 9e9, bw=1e9, label='bp2'),
])

# Four-way multiplexer
mux = design_multiplexer([
    FilterSpec('lp', 5, 5e9),
    FilterSpec('bp', 5, 7e9, bw=1e9, label='bp1'),
    FilterSpec('bp', 5, 9e9, bw=1e9, label='bp2'),
    FilterSpec('hp', 5, 11e9),
])
```

## Netlist Generation

### filter_to_netlist

Convert a FilterDesign into a JC-compatible 2-port netlist.

```python
netlist = filter_to_netlist(design, prefix='f1')
# netlist.port_map = {'input': (1, 'f1_in'), 'output': (2, 'f1_out')}
```

### multiplexer_to_netlist

Convert a MultiplexerDesign into an N-port netlist. Port 1 is the common port; subsequent ports correspond to each arm in the order specified in `arm_specs`.

```python
netlist = multiplexer_to_netlist(mux, prefix='m1')
# Diplexer: port_map = {'common': (1, 'm1_c'), 'lp': (2, ...), 'hp': (3, ...)}
```

**Node naming convention:**
- Common node: `{prefix}_c`
- Arm internal nodes: `{prefix}_{label}_1`, `{prefix}_{label}_2`, ...
- Ground: `0`

## Analysis

### peripheral_response

Calculate S-parameter response of a peripheral circuit standalone.

```python
resp = peripheral_response(design, f=np.linspace(1, 15, 1001), units='GHz')
```

### plot_peripheral_response

Plot the frequency response. Accepts FilterDesign, MultiplexerDesign, or raw transfo_result dict.

```python
fig, axes, resp = plot_peripheral_response(mux, units='GHz')
```

## Topology Composition

### compose_chain

Compose an arbitrary chain of peripherals and TWPAs into a single netlist.

```python
comps, params, meta = compose_chain(
    blocks=[diplexer_in, twpa_dict, diplexer_out],
    connections=[
        ('common', 'input'),   # diplexer_in common -> TWPA port 1
        ('output', 'common'),  # TWPA port 2 -> diplexer_out common
    ],
)
```

**Parameters:**
- `blocks`: List of PeripheralNetlist and/or TWPA netlist dicts (with keys `jc_components`, `circuit_parameters`, `metadata`)
- `connections`: List of `(out_role, in_role)` tuples specifying how adjacent blocks connect. The roles come from each block's port_map keys.
- `Z0`: Reference impedance (default 50)

**How it works:**
1. Strips all ports and port-termination resistors from every block
2. Merges nodes at junctions (so connected blocks share a node)
3. Concatenates all components in logical order (input peripherals read port-to-junction, TWPA in natural order, output peripherals read junction-to-port)
4. Adds fresh ports and resistors at the surviving external nodes with sequential numbering

**Port numbering:** External ports are numbered sequentially by block order, then by arm order within each block. The arm order matches the `arm_specs` list order in `design_multiplexer`.

### stitch_filter_twpa_filter

Convenience wrapper for filter -> TWPA -> filter topology.

```python
comps, params, meta = stitch_filter_twpa_filter(input_filter, twpa_dict, output_filter)
```

Either filter can be `None`.

### stitch_diplexer_twpa_chain

Convenience wrapper for the TWPA-TWPA cascade: diplexer -> TWPA -> diplexer -> TWPA -> ... -> diplexer.

```python
comps, params, meta = stitch_diplexer_twpa_chain(
    diplexers=[d1, d2, d3],  # N+1 diplexers
    twpas=[twpa1, twpa2],    # N TWPAs
)
```

### save_peripheral_netlist

Save composed netlist to a .py file readable by `julia_wrapper.load_netlist()`.

```python
save_peripheral_netlist(comps, params, meta, "netlists/my_device.py")
```

## Supported Filter Types

| Type | response | Description |
|------|----------|-------------|
| Low-pass | `'lp'` | Passes frequencies below fc |
| High-pass | `'hp'` | Passes frequencies above fc |
| Band-pass | `'bp'` | Passes frequencies around fc (requires `bw`) |
| Band-stop | `'bs'` | Rejects frequencies around fc (requires `bw`) |

## Supported Approximations

| Approximation | approx | Parameters |
|---------------|--------|------------|
| Butterworth | `'butterworth'` | Maximally flat passband |
| Chebyshev Type I | `'chebyshev1'` | Equiripple passband (requires `ripple_dB`) |

## Example: Diplexed TWPA

See `examples/diplexer_twpa_example.py` for a complete workflow that:
1. Designs LP/HP diplexers
2. Composes them with a TWPA netlist into a 4-port device
3. Simulates with JosephsonCircuits.jl
4. Plots S31 (signal gain), S13, S42 (pump path), S11, S33 (reflections), and quantum efficiency

```python
from twpa_design.filter_builder import *

# Design diplexers
mux = design_multiplexer([
    FilterSpec('lp', 25, 8.3e9, label='lp'),
    FilterSpec('hp', 25, 8.3e9, label='hp'),
], Z0=50.0)

# Generate netlists
d_in = multiplexer_to_netlist(mux, prefix='m1')
d_out = multiplexer_to_netlist(mux, prefix='m2')

# Compose: diplexer -> TWPA -> diplexer
comps, params, meta = compose_chain(
    [d_in, twpa_dict, d_out],
    connections=[('common', 'input'), ('output', 'common')],
)

# Save and simulate
save_peripheral_netlist(comps, params, meta, "diplexed_twpa.py")

# After simulation, access any S-parameter:
results.s_param(3, 1)        # S31: signal gain (LP in -> LP out)
results.s_param(4, 2)        # S42: pump transmission (HP in -> HP out)
results.s_harmonic(1, 4, 1)  # Idler: mode 1, HP out, LP in
```

## g-value Computation

The module includes a high-precision g-value calculator (using `mpmath` with 50 decimal places) supporting:
- Butterworth and Chebyshev Type I prototypes
- Doubly terminated (standalone filters) and singly terminated (multiplexer arms)
- Even-order Chebyshev modification

```python
from twpa_design.filter_builder import calculate_g_values

g = calculate_g_values('butterworth', order=5, termination='double')
# [1.0, 0.618, 1.618, 2.0, 1.618, 0.618, 1.0]

g = calculate_g_values('chebyshev1', order=5, ripple_dB=0.5, termination='single')
```
