# twpa_design - Engineering Notes

Living document tracking design decisions, physical reasoning, and implementation notes for the `twpa_design` package.

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
