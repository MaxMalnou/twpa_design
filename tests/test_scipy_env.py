"""Env smoke test for scipy's Fortran L-BFGS-B optimizer.

A BLAS/LAPACK DLL mismatch (typically: pip-installed scipy on top of a
conda env, or vice versa) causes Windows fatal exception 0xc06d007f when
scipy's L-BFGS-B is called. That crash is raised by the OS below the
Python interpreter, so it cannot be caught with try/except — it kills the
whole process. We therefore run the smoke check in a subprocess and
inspect the exit code.

The package's phase-matching code calls L-BFGS-B via basin-hopping in
helper_functions.find_all_solutions and the corresponding refinement step,
so a broken env would silently kill any design run that uses
stopbands_config_GHz. See the Troubleshooting section in README.md.
"""
import subprocess
import sys
import textwrap


def test_scipy_lbfgsb_runs():
    script = textwrap.dedent("""
        import numpy as np
        from scipy.optimize import minimize, basinhopping

        # Mirror the helper_functions usage: bounded L-BFGS-B inside basin-hopping
        # over a quartic-ish objective.
        def obj(x):
            return float((x[0] - 0.123) ** 2 * (1 + np.sin(3 * x[0]) ** 2))

        def take_step(x):
            return np.clip(x + np.random.uniform(-0.1, 0.1, len(x)), -0.5, 0.5)

        res = minimize(obj, [-0.3], method='L-BFGS-B', bounds=[(-0.5, 0.5)],
                       options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1000})
        assert res.fun < 1e-6, f'L-BFGS-B minimize did not converge: {res.fun}'

        mk = {'method': 'L-BFGS-B', 'bounds': [(-0.5, 0.5)],
              'options': {'ftol': 1e-7, 'gtol': 1e-7, 'maxiter': 1000}}
        res = basinhopping(obj, [-0.5], minimizer_kwargs=mk, niter=5, T=0.5,
                           stepsize=0.05, take_step=take_step, interval=10)
        assert res.fun < 1e-4, f'L-BFGS-B basinhopping did not converge: {res.fun}'
    """)
    result = subprocess.run(
        [sys.executable, '-X', 'faulthandler', '-c', script],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise AssertionError(
            "scipy's L-BFGS-B crashed in this env (likely BLAS/LAPACK DLL "
            "mismatch from mixing pip and conda installs of numpy/scipy). "
            "See README.md Troubleshooting section.\n\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
