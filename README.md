# TWPA Design Package

A comprehensive Python package for designing and simulating Traveling Wave Parametric Amplifiers (TWPAs) using the Artificial Transmission Line (ATL) methodology and JosephsonCircuits.jl.

## Features

The package provides three main modules:

- **🎯 ATL TWPA Designer**: Design TWPAs with engineered dispersion through filters and/or periodic modulation
- **⚡ Netlist Builder**: Convert TWPA designs into circuit netlists compatible with JosephsonCircuits.jl
- **🧪 Julia Wrapper**: Run harmonic balance simulations via JosephsonCircuits.jl from Python

## Installation

### Requirements

- **Python** ≥ 3.8
- **Julia** ≥ 1.6 (download from [julialang.org](https://julialang.org/downloads/))

### Python Dependencies

The package requires the following Python libraries:

```bash
pip install numpy>=2.0 matplotlib>=3.5 julia>=0.6 scipy>=1.7 mpmath>=1.2
```

**Note:** This package requires NumPy 2.0 or later. The code has been updated to be compatible with NumPy 2.x's stricter array truthiness checks.

### Install Package

1. **Direct install from GitHub** (easiest):
   ```bash
   pip install git+https://github.com/MaxMalnou/twpa_design.git
   ```

2. **Clone the repository** (recommended for full examples and notebooks):
   ```bash
   git clone --recursive https://github.com/MaxMalnou/twpa_design.git
   cd twpa_design
   pip install -e .
   ```
   
   If you already cloned without `--recursive`, run:
   ```bash
   git submodule update --init --recursive
   ```

3. **Or install from PyPI** (when available):
   ```bash
   pip install twpa_design
   ```

### Julia Setup

The package automatically sets up the required Julia packages on first use. By default, it uses a GitHub fork of JosephsonCircuits.jl with Taylor expansion support for nonlinear elements:

```python
from twpa_design.julia_wrapper import TWPASimulator
simulator = TWPASimulator()  # Auto-installs Julia dependencies
```

**Configuration:** Julia package sources are controlled in `julia_setup.py`:
- `USE_LOCAL_FORK = False`: Use remote version (default for users)
- `USE_GITHUB_FORK = True`: Use GitHub fork with Taylor expansion features (default)

## Quick Start

### 1. Design a TWPA

```python
from twpa_design.atl_twpa_designer import ATLTWPADesigner
import numpy as np

# Design a 4WM JTWPA with filter dispersion
designer = ATLTWPADesigner(
    custom_params={
        'f_zeros_GHz': 9,          # Single number (converted automatically)
        'f_poles_GHz': 8.85,       # Single number (converted automatically)  
        'Ic_JJ_uA': 5,
        'Ia0_uA': 3,
        'Ntot_cell': 2000
    }
)

results = designer.run_design(interactive=True, save_results=True)
```

### 2. Generate Circuit Netlist

```python
from twpa_design.netlist_JC_builder import NetlistConfig, build_netlist_from_config

config = NetlistConfig(design_file='4wm_jtwpa_01.py')
netlist_file = build_netlist_from_config(config)
```

### 3. Run Simulation

```python
from twpa_design.julia_wrapper import TWPASimulator, TWPASimulationConfig

simulator = TWPASimulator()
results = simulator.run_full_simulation(
    netlist_name="4wm_jtwpa_2002cells_01",
    config=TWPASimulationConfig(
        freq_start_GHz=6.0,
        freq_stop_GHz=10.0,
        pump_freq_GHz=8.63,
        pump_current_A=2.7e-6
    ),
    save_results=True,
    show_plot=True
)
```

## Typical Workflow

```
Design TWPA → Generate Netlist → Run Simulation → Analyze Results
    ↓              ↓                 ↓               ↓
ATL Designer → Netlist Builder → Julia Wrapper → Results Analysis
```

1. **Design**: Use `ATLTWPADesigner` to optimize TWPA parameters and find phase-matched pump frequency
2. **Netlist**: Use `NetlistConfig` to convert design into JosephsonCircuits.jl-compatible circuit
3. **Simulate**: Use `TWPASimulator` to run harmonic balance analysis and calculate S-parameters
4. **Analyze**: Built-in plotting and analysis tools for gain, bandwidth, and quantum efficiency

## Documentation

- **[ATL TWPA Designer Reference](docs/atl_twpa_designer_reference.md)** - Complete guide to TWPA design
- **[Netlist Builder Reference](docs/netlist_JC_builder_reference.md)** - Circuit netlist generation
- **[Julia Wrapper Reference](docs/julia_wrapper_reference.md)** - Harmonic balance simulations

## Package Structure

```
twpa_design/
├── README.md
├── pyproject.toml
├── docs/                           # Detailed documentation
│   ├── atl_twpa_designer_reference.md
│   ├── netlist_JC_builder_reference.md
│   └── julia_wrapper_reference.md
└── src/twpa_design/
    ├── __init__.py
    ├── atl_twpa_designer.py        # TWPA design module
    ├── netlist_JC_builder.py       # Circuit netlist generator
    ├── julia_wrapper.py            # JosephsonCircuits.jl interface
    ├── julia_setup.py              # Julia environment setup
    ├── helper_functions.py         # Utility functions
    ├── plots_params.py             # Plotting configuration
    ├── designs/                    # Example TWPA designs
    │   ├── 4wm_jtwpa_01.py
    │   ├── 4wm_ktwpa_01.py
    │   └── b_jtwpa_01.py
    ├── examples/                   # Usage examples
    │   ├── atl_twpa_designer_example.py
    │   ├── atl_twpa_plotter_example.py
    │   ├── netlist_JC_builder_example.py
    │   └── julia_wrapper_example.py
    ├── netlists/                   # Example circuit netlists
    │   ├── 4wm_jtwpa_2002cells_01.py
    │   ├── 4wm_ktwpa_5004cells_01.py
    │   └── b_jtwpa_2000cells_01.py
    ├── notebooks/                  # Jupyter tutorials
    │   ├── atl_twpa_designer.ipynb
    │   ├── julia_wrapper.ipynb
    │   ├── julia_wrapper_basic_tests.ipynb
    │   ├── JJvsNL_comparison.ipynb
    │   └── netlist_JC_builder.ipynb
    ├── results/                    # Example simulation results
    │   ├── 4wm_jtwpa_2002cells_01_pump8.63GHz_01.npz
    │   ├── 4wm_jtwpa_2002cells_01_pump8.63GHz_01.svg
    │   ├── 4wm_ktwpa_5004cells_01_pump9.10GHz_01.npz
    │   ├── 4wm_ktwpa_5004cells_01_pump9.10GHz_01.svg
    │   ├── b_jtwpa_2000cells_01_pump16.12GHz_01.npz
    │   └── b_jtwpa_2000cells_01_pump16.12GHz_01.svg
    └── external_packages/          # JosephsonCircuits.jl fork
        └── JosephsonCircuits.jl/   # Fork with Taylor expansion NL elements
            ├── README.md           # Fork documentation and usage
            ├── examples/           # JJ vs NL comparison examples
            └── src/                # Modified Julia source code
```

## Examples and Notebooks

The package includes comprehensive examples:

- **Design examples**: `/examples/` - Standalone Python scripts
- **Interactive tutorials**: `/notebooks/` - Jupyter notebooks with step-by-step guides
- **Pre-made designs**: `/designs/` - Ready-to-use TWPA configurations
- **Sample netlists**: `/netlists/` - Example circuit files
- **Reference results**: `/results/` - Sample simulation outputs

To explore the notebooks:
```bash
cd src/twpa_design/notebooks
jupyter lab
```

## Supported TWPA Types

- **JTWPA**: Josephson junction TWPAs (3WM and 4WM operation)
- **KTWPA**: Kinetic inductance TWPAs (3WM and 4WM operation)

## Dispersion Engineering

- **Filter dispersion**: Foster form L/C filters at zero/pole frequencies
- **Periodic modulation**: Spatial modulation with windowing (boxcar, Tukey, Hann)
- **Combined approach**: Both filter and periodic dispersion

## Supported JJ Structures

- **JJ**: Standard Josephson junctions
- **RF-SQUID**: Flux-tunable RF-biased SQUIDs with Kerr-free operation

## Nonlinear Elements

Two modeling approaches:

1. **Josephson Junctions**: Full JJ potential using JosephsonCircuits.jl's hardcoded implementation
2. **Taylor Expansion**: General nonlinearity expanded up to 4th order (c1, c2, c3, c4 terms)
   - Can model JJs, RF-SQUIDs, and kinetic inductance
   - KI: L(I) = L₀(1 + (I/I*)²) expanded to 2nd order (c1, c2 only)

## Future Improvements

- ⭕ Netlist visualization
- ⭕ Systematic workflow for multi-mode resonator-based parametric amplifiers

## License

MIT License - see [LICENSE](LICENSE) file for details

## Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

### How to Contribute

1. **Report Issues**: Open an issue describing bugs or feature requests
2. **Submit Pull Requests**: 
   - Fork the repository
   - Create a feature branch (`git checkout -b feature/new-feature`)
   - Commit changes with descriptive messages
   - Push to your fork and submit a pull request

### Development Guidelines

- Follow existing code style and conventions
- Add tests for new functionality when applicable
- Update documentation for significant changes
- Ensure all tests pass before submitting

### Questions?

Open an issue for questions about contributing or using the package.

## Citation

If you use this package in your research, please cite:

```
Maxime Malnou, "TWPA Design Package: Python tools for designing traveling wave parametric amplifiers," 2025. 
Available: https://github.com/MaxMalnou/twpa_design
```

### BibTeX
```bibtex
@software{twpa_design_2025,
  author = {Maxime Malnou},
  title = {TWPA Design Package: Python tools for designing traveling wave parametric amplifiers},
  year = {2025},
  url = {https://github.com/MaxMalnou/twpa_design}
}
```

### Related Work
If you use specific methodologies implemented in this package, please also cite the relevant research papers that describe the underlying theory.

## Acknowledgments

Development of this package was assisted by [Claude.ai](https://claude.ai) and [Claude Code](https://claude.ai/code) for code generation, debugging, and documentation.

## Contact

For questions or collaboration opportunities, please [open an issue](https://github.com/MaxMalnou/twpa_design/issues).