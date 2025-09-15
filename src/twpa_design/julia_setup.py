"""
Simplified Julia setup for TWPA package with fork environment support.

For Windows users with juliaup:
If Julia is not found automatically, set the JULIA_BASE environment variable:
  set JULIA_BASE=C:\\Users\\YourName\\.julia\\juliaup\\julia-1.10.0+0.x64.w64.mingw32

Or add to your system environment variables for permanent configuration.
"""

import os
import sys
import shutil
from pathlib import Path
from typing import Optional, Any, Tuple, Dict
import contextlib
import io


# ============================================================================
# JosephsonCircuits.jl Configuration
# ============================================================================
# Toggle between different versions of JosephsonCircuits.jl
#
# USE_LOCAL_FORK: True = local development with hot-reloading (Revise.jl)
#                 False = use remote version (GitHub or registered)
#
# USE_GITHUB_FORK: True = use GitHub PR branch (MaxMalnou fork)
#                  False = use Kevin's registered version
#                  (Only applies when USE_LOCAL_FORK=False)
# ============================================================================

USE_LOCAL_FORK = False  # Set to True for local development
USE_GITHUB_FORK = True  # Set to False after PR is merged to Kevin's repo

GITHUB_FORK_URL = "https://github.com/MaxMalnou/JosephsonCircuits.jl"
GITHUB_FORK_BRANCH = "taylor-expansion-feature"

# ============================================================================
# Internal constant - do not modify
# ============================================================================
# JosephsonCircuits.jl package UUID (assigned by Julia registry, never changes)
_JOSEPHSON_CIRCUITS_UUID = "23a5dba6-321f-4fcf-be1e-689e290df087"

# ============================================================================

_julia_session = None
_julia_initialized = False


def find_package_root():
    """Find the twpa_design package root directory."""
    # Get the directory where this julia_setup.py file is located
    current_file = Path(__file__).resolve()
    # Navigate: julia_setup.py is in twpa_design/src/twpa_design/
    # So current_file.parent is already the package root: twpa_design/src/twpa_design/
    package_root = current_file.parent
    return package_root


def check_package_source(jl) -> str:
    """Check which version of JosephsonCircuits is loaded.

    Helper function for notebooks to determine package source without hardcoding UUIDs.

    Args:
        jl: Julia instance (from get_julia_for_session())

    Returns:
        "local" if using local fork with hot-reloading
        "github" if using GitHub fork
        "registered" if using registered version
        "unknown" if cannot determine
    """
    try:
        # Check the loaded path first
        pkg_path = str(jl.eval("pathof(JosephsonCircuits)"))

        if 'external_packages' in pkg_path:
            return "local"

        # Check git_source in package metadata
        is_github_fork = jl.eval(f"""
            import Pkg
            deps = Pkg.dependencies()
            jc_uuid = Base.UUID("{_JOSEPHSON_CIRCUITS_UUID}")
            if haskey(deps, jc_uuid)
                dep_info = deps[jc_uuid]
                dep_info.is_tracking_repo && !isnothing(dep_info.git_source) && occursin("MaxMalnou", dep_info.git_source)
            else
                false
            end
        """)

        return "github" if is_github_fork else "registered"
    except:
        return "unknown"


def find_julia_base():
    """Find Julia base directory (not bin) with user configuration support."""
    # Option 1: User-specified JULIA_BASE environment variable
    if "JULIA_BASE" in os.environ:
        return os.environ["JULIA_BASE"]
    
    # Option 2: Derive from JULIA_BINDIR if set
    if "JULIA_BINDIR" in os.environ:
        return os.path.dirname(os.environ["JULIA_BINDIR"])
    
    # Option 3: Check common locations for juliaup
    if sys.platform == "win32":
        # Default juliaup location
        juliaup_base = os.path.expanduser(r"~\.julia\juliaup")
        if os.path.exists(juliaup_base):
            # Look for julia installations
            for item in os.listdir(juliaup_base):
                if item.startswith("julia-") and "mingw" in item:
                    julia_base = os.path.join(juliaup_base, item)
                    julia_exe = os.path.join(julia_base, "bin", "julia.exe")
                    if os.path.exists(julia_exe):
                        return julia_base
        
        # Try standard Program Files locations
        program_locations = [
            os.path.expanduser(r"~\AppData\Local\Programs"),
            r"C:\Program Files",
            r"C:\Program Files (x86)"
        ]
        for loc in program_locations:
            if os.path.exists(loc):
                for item in os.listdir(loc):
                    if item.startswith("Julia"):
                        julia_base = os.path.join(loc, item)
                        julia_exe = os.path.join(julia_base, "bin", "julia.exe")
                        if os.path.exists(julia_exe):
                            return julia_base
    
    # Option 4: Try to find from PATH
    import shutil
    julia_path = shutil.which("julia")
    if julia_path:
        # Go up from julia executable to base
        julia_bin = os.path.dirname(julia_path)
        return os.path.dirname(julia_bin)
    
    return None


class JuliaEnvironment:
    """Context manager for temporary Julia environment setup"""
    
    def __init__(self):
        self.original_env = {}
        self.julia_configured = False
        
    def __enter__(self):
        """Set up Julia environment temporarily"""
        # Save original environment
        self.original_env = {
            'PATH': os.environ.get('PATH', ''),
            'JULIA_BINDIR': os.environ.get('JULIA_BINDIR', ''),
            'JULIA': os.environ.get('JULIA', '')
        }
        
        # Configure for PyCall
        self._configure_for_pycall()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Restore original environment"""
        for key, value in self.original_env.items():
            if value:
                os.environ[key] = value
            elif key in os.environ:
                del os.environ[key]
                
    def _configure_for_pycall(self):
        """Configure environment for PyCall compatibility"""
        # Find Julia base directory
        julia_base = find_julia_base()
        
        if not julia_base:
            raise RuntimeError(
                "Could not find Julia installation. Please either:\n"
                "1. Set JULIA_BASE environment variable to your Julia directory\n"
                "   Example (Windows): set JULIA_BASE=C:\\Users\\YourName\\.julia\\juliaup\\julia-1.10.0+0.x64.w64.mingw32\n"
                "2. Set JULIA_BINDIR to your Julia bin directory\n"
                "3. Add Julia to your PATH"
            )
        
        # Set up paths (same logic as original julia_setup_improved.py)
        julia_bin_path = os.path.join(julia_base, "bin")
        julia_lib_path = os.path.join(julia_base, "lib")
        julia_exe = os.path.join(julia_bin_path, "julia.exe" if sys.platform == "win32" else "julia")
        
        # Verify Julia executable exists
        if not os.path.exists(julia_exe):
            raise RuntimeError(f"Julia executable not found at: {julia_exe}")
        
        # Add to PATH for this session only (critical for PyCall on Windows)
        current_path = os.environ.get("PATH", "")
        path_sep = ";" if sys.platform == "win32" else ":"
        os.environ["PATH"] = f"{julia_bin_path}{path_sep}{julia_lib_path}{path_sep}{current_path}"
        os.environ["JULIA_BINDIR"] = julia_bin_path
        os.environ["JULIA"] = julia_exe
        
        print(f"📍 Using Julia from: {julia_base}")


def get_julia_for_session(use_local_josephson: Optional[bool] = None,
                         local_path: Optional[str] = None,
                         enable_revise: bool = True,
                         force_reinit: bool = False) -> Any:
    """Get a Julia instance with proper environment setup.

    Args:
        use_local_josephson: Override USE_LOCAL_FORK setting. If None, uses global config.
        local_path: Path to local fork (only used if use_local_josephson=True)
        enable_revise: Enable Revise.jl for hot-reloading
        force_reinit: Force reinitialization even if session exists
    """
    global _julia_session, _julia_initialized

    # Use global configuration if not explicitly overridden
    if use_local_josephson is None:
        use_local_josephson = USE_LOCAL_FORK
    
    # Check for forced reinit from environment variable
    force_reinit = force_reinit or os.environ.get("TWPA_FORCE_JULIA_REINIT", "0") == "1"
    if force_reinit and "TWPA_FORCE_JULIA_REINIT" in os.environ:
        del os.environ["TWPA_FORCE_JULIA_REINIT"]  # Clear the flag
    
    # Check if we already have a working session in this Python process
    if _julia_initialized and _julia_session is not None and not force_reinit:
        try:
            _julia_session.eval("1+1")  # Test if it's alive
            print("📌 Using existing Julia session")
            return _julia_session
        except:
            # Session died, need to reinitialize
            _julia_initialized = False
            _julia_session = None
    
    # Find and configure Julia paths
    julia_base = find_julia_base()
    if not julia_base:
        raise RuntimeError(
            "Could not find Julia installation. Please either:\n"
            "1. Set JULIA_BASE environment variable to your Julia directory\n"
            "2. Add Julia to your PATH"
        )

    # Check for Julia version upgrade BEFORE initializing PyJulia
    print("🚀 Initializing new Julia session...")
    print(f"📍 Using Julia from: {julia_base}")

    import subprocess
    import re
    try:
        # Properly handle the julia executable path
        if os.path.isfile(julia_base):
            julia_exe = julia_base
        else:
            if os.path.exists(os.path.join(julia_base, 'bin', 'julia.exe')):
                julia_exe = os.path.join(julia_base, 'bin', 'julia.exe')
            else:
                julia_exe = os.path.join(julia_base, 'julia.exe' if os.name == 'nt' else 'julia')

        result = subprocess.run([julia_exe, '--version'], capture_output=True, text=True)
        julia_version = result.stdout.strip()
        print(f"Julia version: {julia_version}")

        # Extract version number
        version_match = re.search(r'(\d+\.\d+\.\d+)', julia_version)
        if version_match:
            current_version = version_match.group(1)

            # Look for newer versions
            if 'juliaup' in julia_exe:
                parts = julia_exe.replace('\\', '/').split('/')
                juliaup_idx = parts.index('juliaup')
                juliaup_dir = '/'.join(parts[:juliaup_idx+1]).replace('/', os.sep)

                if os.path.exists(juliaup_dir):
                    julia_dirs = [d for d in os.listdir(juliaup_dir)
                                 if d.startswith('julia-') and os.path.isdir(os.path.join(juliaup_dir, d))]

                    versions = []
                    for julia_dir in julia_dirs:
                        ver_match = re.search(r'julia-(\d+\.\d+\.\d+)', julia_dir)
                        if ver_match:
                            versions.append((ver_match.group(1), julia_dir))

                    if versions:
                        versions.sort(key=lambda x: [int(i) for i in x[0].split('.')])
                        latest_version, latest_dir = versions[-1]

                        if latest_version != current_version:
                            latest_julia_base = os.path.join(juliaup_dir, latest_dir)
                            latest_julia_exe = os.path.join(latest_julia_base, 'bin', 'julia.exe')
                            if os.path.exists(latest_julia_exe):
                                julia_base = latest_julia_base
                                julia_exe = latest_julia_exe
                                print(f"⬆️  Upgraded to Julia {latest_version}")
    except Exception as e:
        print(f"Version check note: {e}")
        pass

    # Set up paths (using potentially upgraded julia_base)
    julia_bin_path = os.path.join(julia_base, "bin")
    julia_lib_path = os.path.join(julia_base, "lib")
    julia_exe = os.path.join(julia_bin_path, "julia.exe" if sys.platform == "win32" else "julia")

    # Modify PATH for this Python session
    current_path = os.environ.get("PATH", "")
    path_sep = ";" if sys.platform == "win32" else ":"
    os.environ["PATH"] = f"{julia_bin_path}{path_sep}{julia_lib_path}{path_sep}{current_path}"
    os.environ["JULIA_BINDIR"] = julia_bin_path
    os.environ["JULIA"] = julia_exe
    
    # Set Julia startup options to suppress banner
    os.environ["PYJULIA_JULIA_STARTUP_ARGS"] = "--banner=no --quiet"
    
    # Now import and configure julia
    try:
        import julia
        julia.install()
        from julia import Main as jl  # type: ignore
    except Exception as e:
        if "libpython" in str(e):
            import julia
            julia.install()
            from julia import Main as jl  # type: ignore
        else:
            raise RuntimeError(f"Failed to initialize Julia: {e}")
    
    # CHECK IF FORK IS ALREADY SET UP PROPERLY
    if use_local_josephson and not force_reinit:
        try:
            # Check if JosephsonCircuits is installed as dev package from our fork
            jl.eval("import Pkg")
            
            # Check package status
            is_dev_installed = jl.eval(f"""
                deps = Pkg.dependencies()
                jc_uuid = Base.UUID("{_JOSEPHSON_CIRCUITS_UUID}")
                
                if haskey(deps, jc_uuid)
                    dep_info = deps[jc_uuid]
                    # Check if it's a dev package and from our fork location
                    if dep_info.is_tracking_path && occursin("external_packages", dep_info.source)
                        true
                    else
                        false
                    end
                else
                    false
                end
            """)
            
            if is_dev_installed:
                print("📌 Fork already installed as dev package")
                
                # Load it
                jl.eval("using JosephsonCircuits")
                pkg_path = jl.eval('pathof(JosephsonCircuits)')
                print(f"  📍 Loaded from: {pkg_path}")
                
                # Ensure Revise is set up
                if enable_revise:
                    if not jl.eval("isdefined(Main, :Revise)"):
                        print("🔧 Loading Revise.jl...")
                        jl.eval("using Revise")
                        print("✅ Revise.jl loaded")
                    print("✅ Hot-reloading active - edit .jl files and changes auto-reload!")
                
                # Cache and return - skip all the cleanup!
                _julia_session = jl
                _julia_initialized = True
                return jl
                
        except Exception as e:
            print(f"Fork check failed: {e}")
            pass  # If check fails, do full setup
    
    # If using fork and no path specified, use the bundled one
    if use_local_josephson and local_path is None:
        package_root = find_package_root()
        local_path = os.path.join(package_root, "external_packages", "JosephsonCircuits.jl")
    
    # Set up Revise if requested (MUST be done before loading packages)
    if enable_revise and use_local_josephson:
        print("🔧 Setting up Revise.jl for automatic code reloading...")
        
        # Install Revise if not already installed
        jl.eval("""
            import Pkg
            if !haskey(Pkg.dependencies(), Base.UUID("295af30f-e4ad-537b-8983-00126c2a3abe"))
                Pkg.add("Revise"; io=devnull)
            end
        """)
        
        # Load Revise
        jl.eval("using Revise")
        print("✅ Revise.jl loaded - changes will be tracked automatically")
    
    # If using local fork, set it up
    if use_local_josephson:
        assert local_path is not None  # Tell Pylance it's not None
        
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local fork not found at: {local_path}")
        
        # Convert Windows path to Julia-friendly format
        local_path_julia = local_path.replace('\\', '/')
        
        print(f"🔧 Setting up local fork from: {local_path}")
        
        # First ensure Pkg is available
        jl.eval("import Pkg")
        
        # Always do full cleanup when initializing fresh
        print("  📦 Cleaning up existing installations...")
        try:
            jl.eval('Pkg.rm("JosephsonCircuits")')
            print("  ✓ Removed registered version")
        except:
            print("  ℹ️ No registered version to remove")
        
        try:
            jl.eval('Pkg.free("JosephsonCircuits")')
            print("  ✓ Freed existing dev version")
        except:
            print("  ℹ️ No dev version to free")
        
        # Add the fork
        print("  🔧 Adding fork as development package...")
        jl.eval(f'Pkg.develop(path="{local_path_julia}")')
        
        # Check where it will load from
        print("  🔍 Checking package location...")
        pkg_path = jl.eval('Base.find_package("JosephsonCircuits")')
        print(f"  📍 Package will load from: {pkg_path}")
        
        if pkg_path and "external_packages" not in pkg_path:
            print("  ❌ WARNING: Not loading from fork!")
            print("  🔄 Trying to force reload...")
            
            # Try to force it
            jl.eval("Pkg.precompile()")
            pkg_path = jl.eval('Base.find_package("JosephsonCircuits")')
            print(f"  📍 After precompile, loads from: {pkg_path}")
        
        if enable_revise:
            print("💡 Any changes to .jl files will be automatically reloaded!")
    
    else:
        # Use remote version (GitHub fork or registered)
        if USE_GITHUB_FORK:
            print(f"📦 Using GitHub fork: {GITHUB_FORK_URL}")
            print(f"   Branch: {GITHUB_FORK_BRANCH}")

            jl.eval("import Pkg")

            # Check if fork is already installed in the CURRENT active environment
            print("  🔍 Checking if GitHub fork is already installed...")

            # First ensure we're checking the default environment (not a project-specific one)
            jl.eval('import Pkg; Pkg.activate()')

            # Check if fork is already installed
            is_fork_installed = jl.eval(f"""
                import Pkg
                deps = Pkg.dependencies()
                jc_uuid = Base.UUID("{_JOSEPHSON_CIRCUITS_UUID}")

                if haskey(deps, jc_uuid)
                    dep_info = deps[jc_uuid]
                    # Check if it's from our GitHub fork by checking git_source
                    if dep_info.is_tracking_repo && !isnothing(dep_info.git_source)
                        if occursin("MaxMalnou", dep_info.git_source)
                            true
                        else
                            false
                        end
                    else
                        false
                    end
                else
                    false
                end
            """)

            if is_fork_installed:
                print("  ✅ GitHub fork already installed, skipping clone")
            else:
                print("  📦 GitHub fork not found, installing...")

                # Fix SSL certificate issue on Windows
                print("  🔧 Configuring SSL certificates for GitHub access...")
                try:
                    jl.eval('ENV["JULIA_SSL_CA_ROOTS_PATH"] = ""')
                    print("  ✓ SSL configuration set")
                except:
                    pass  # Continue even if this fails

                # Remove any existing versions
                print("  📦 Cleaning up existing installations...")
                try:
                    jl.eval('Pkg.rm("JosephsonCircuits")')
                    print("  ✓ Removed existing version")
                except:
                    print("  ℹ️ No existing version to remove")

                try:
                    jl.eval('Pkg.free("JosephsonCircuits")')
                    print("  ✓ Freed existing dev version")
                except:
                    print("  ℹ️ No dev version to free")

                # Add from GitHub
                print("  🔧 Cloning from GitHub fork...")
                jl.eval(f'Pkg.add(url="{GITHUB_FORK_URL}", rev="{GITHUB_FORK_BRANCH}")')
                print("  ✅ Clone completed successfully")

            # Load the package
            print("  📦 Loading JosephsonCircuits...")
            jl.eval('using JosephsonCircuits')

            # Verify by checking the git source
            print("  🔍 Verifying installation...")
            pkg_path = jl.eval('pathof(JosephsonCircuits)')
            print(f"  📍 Package loaded from: {pkg_path}")

            # Verification: check the git source in package metadata
            try:
                is_our_fork = jl.eval(f"""
                    import Pkg
                    deps = Pkg.dependencies()
                    jc_uuid = Base.UUID("{_JOSEPHSON_CIRCUITS_UUID}")
                    if haskey(deps, jc_uuid)
                        dep_info = deps[jc_uuid]
                        dep_info.is_tracking_repo && !isnothing(dep_info.git_source) && occursin("MaxMalnou", dep_info.git_source)
                    else
                        false
                    end
                """)

                if is_our_fork:
                    print("  ✅ Successfully using GitHub fork!")
                else:
                    print("  ⚠️ Warning: May not be using the fork")
            except:
                # Fallback: just check if it was installed (it worked)
                print("  ✅ Package installed and loaded")

        else:
            # Use registered version (after PR merge)
            print("📦 Using registered JosephsonCircuits.jl")
            jl.eval(f"""
                import Pkg

                # Remove dev version if it exists
                try
                    Pkg.free("JosephsonCircuits")
                catch
                    # Not a dev package, that's fine
                end

                # Add registered version if not already there
                if !haskey(Pkg.dependencies(), Base.UUID("{_JOSEPHSON_CIRCUITS_UUID}"))
                    Pkg.add("JosephsonCircuits")
                end

                # Load it
                using JosephsonCircuits
            """)
    
    # Cache the session for reuse
    _julia_session = jl
    _julia_initialized = True
    
    return jl

def reset_julia_session():
    """Force a fresh Julia session on next call."""
    global _julia_session, _julia_initialized
    _julia_session = None
    _julia_initialized = False
    
    # Also set a flag to force full reinit on next call
    os.environ["TWPA_FORCE_JULIA_REINIT"] = "1"
    print("🔄 Julia session marked for full reset")


def force_revise_update(jl=None):
    """
    Manually trigger Revise to check for changes.
    Usually not needed as Revise checks automatically, but useful
    when you want to ensure changes are picked up immediately.
    
    Args:
        jl: Julia instance (if None, will get from session)
    """
    global _julia_session
    
    if jl is None:
        if _julia_session is not None:
            jl = _julia_session
        else:
            # Try to import from julia
            try:
                from julia import Main as jl  # type: ignore
            except ImportError:
                raise RuntimeError("No Julia session available. Run get_julia_for_session() first.")
    
    try:
        # Check if Revise is loaded
        if not jl.eval("isdefined(Main, :Revise)"):
            print("⚠️ Revise.jl not loaded. Loading it now...")
            jl.eval("using Revise")
        
        # Force Revise to check for changes
        jl.eval("Revise.revise()")
        print("✅ Forced Revise update complete - changes should be reflected now")
        
        # If using the fork, also trigger recompilation
        try:
            if jl.eval('Base.find_package("JosephsonCircuits")') and \
               "external_packages" in str(jl.eval('pathof(JosephsonCircuits)')):
                jl.eval("using Pkg; Pkg.precompile()")
                print("✅ Also triggered precompilation of JosephsonCircuits fork")
        except:
            pass
            
    except Exception as e:
        print(f"❌ Failed to force Revise update: {e}")
        print("You may need to restart the Julia session with reset_julia_session()")



def switch_to_registered_isolated(jl: Any) -> Dict:
    """Switch to registered version with complete isolation from fork"""
    print("🔄 Switching to registered JosephsonCircuits (isolated)...")
    
    result = jl.eval("""
        import Pkg
        
        # Create a new temporary environment
        temp_dir = mktempdir()
        Pkg.activate(temp_dir)
        
        # Clear the load path to prevent finding the fork
        empty!(LOAD_PATH)
        push!(LOAD_PATH, "@")  # Only current environment
        push!(LOAD_PATH, "@stdlib")  # And standard library
        
        # Now add ONLY the registered version
        Pkg.add("JosephsonCircuits")
        
        # Load it
        using JosephsonCircuits
        
        # Get info
        pkg_path = pathof(JosephsonCircuits)
        deps = Pkg.dependencies()
        jc_uuid = Base.UUID("{_JOSEPHSON_CIRCUITS_UUID}")
        pkg_version = deps[jc_uuid].version
        
        Dict(
            "path" => pkg_path,
            "version" => string(pkg_version),
            "temp_env" => temp_dir
        )
    """)
    
    return result


def clean_package_cache(package_name: str = "JosephsonCircuits", verbose: bool = True):
    """
    Clean Julia precompilation cache for a specific package.
    
    Args:
        package_name: Name of the package to clean
        verbose: Print status messages
    """
    depot = os.environ.get('JULIA_DEPOT_PATH', os.path.join(os.path.expanduser("~"), ".julia"))
    if ';' in depot:
        depot = depot.split(';')[0]
    
    if verbose:
        print(f"🧹 Cleaning cache for {package_name}...")
    
    # Clean compiled directory
    compiled_dir = os.path.join(depot, "compiled")
    cleaned = False
    
    if os.path.exists(compiled_dir):
        for version_dir in os.listdir(compiled_dir):
            pkg_dir = os.path.join(compiled_dir, version_dir, package_name)
            if os.path.exists(pkg_dir):
                try:
                    shutil.rmtree(pkg_dir)
                    if verbose:
                        print(f"  ✓ Removed cache: {pkg_dir}")
                    cleaned = True
                except Exception as e:
                    if verbose:
                        print(f"  ⚠️ Could not remove {pkg_dir}: {e}")
    
    if verbose:
        if cleaned:
            print("  ✓ Cache cleaning completed")
        else:
            print("  ℹ️ No cache found to clean")
    
    return cleaned


def check_julia_setup() -> bool:
    """
    Quick check if Julia is properly set up.
    
    Returns:
        True if Julia is working, False otherwise
    """
    try:
        # Just check if we can find Julia
        julia_base = find_julia_base()
        return julia_base is not None
    except:
        return False


def get_setup_instructions() -> str:
    """Get platform-specific setup instructions."""
    import sys
    
    instructions = """
Julia Setup Instructions:

1. Install Julia from https://julialang.org/downloads/

2. Add Julia to your system:
"""
    
    if sys.platform == "win32":
        instructions += """
   Windows:
   - Option A: Add Julia to PATH during installation
   - Option B: Set JULIA_BINDIR environment variable:
     setx JULIA_BINDIR "C:\\Users\\YourName\\AppData\\Local\\Programs\\Julia-1.10.0\\bin"
   
3. Install PyJulia:
   pip install julia

4. Initialize PyJulia (one-time setup):
   python -c "import julia; julia.install()"
"""
    elif sys.platform == "darwin":
        instructions += """
   macOS:
   - Option A: Add to PATH in ~/.bash_profile or ~/.zshrc:
     export PATH="/Applications/Julia-1.10.app/Contents/Resources/julia/bin:$PATH"
   - Option B: Set JULIA_BINDIR:
     export JULIA_BINDIR="/Applications/Julia-1.10.app/Contents/Resources/julia/bin"
   
3. Install PyJulia:
   pip install julia

4. Initialize PyJulia (one-time setup):
   python -c "import julia; julia.install()"
"""
    else:  # Linux
        instructions += """
   Linux:
   - Option A: Add to PATH in ~/.bashrc:
     export PATH="/path/to/julia-1.10.0/bin:$PATH"
   - Option B: Set JULIA_BINDIR:
     export JULIA_BINDIR="/path/to/julia-1.10.0/bin"
   
3. Install PyJulia:
   pip install julia

4. Initialize PyJulia (one-time setup):
   python -c "import julia; julia.install()"
"""
    
    return instructions


if __name__ == "__main__":
    """Test Julia setup when run directly."""
    print("Testing Julia setup...")
    print(f"Configuration: USE_LOCAL_FORK={USE_LOCAL_FORK}, USE_GITHUB_FORK={USE_GITHUB_FORK}")

    try:
        jl = get_julia_for_session()  # Use default configuration
        print("✅ Julia setup successful!")

        # Try loading JosephsonCircuits
        try:
            jl.eval("using JosephsonCircuits")
            print("✅ JosephsonCircuits loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load JosephsonCircuits: {e}")

    except Exception as e:
        print(f"❌ Julia setup failed: {e}")
        print("\nSetup instructions:")
        print(get_setup_instructions())