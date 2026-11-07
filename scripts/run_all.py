"""Runs the whole pipeline in order (the weather download, step 2, and the BPDB extension, step 6, need internet
and are already done: their outputs are in data/).

   python scripts/run_all.py --quick   (~25 min: 1 seed, 10 epochs, final pipeline without the live forecast)
   python scripts/run_all.py           (full, ~1.5-2 h on a laptop CPU; reproduces the report numbers)

WARNING: this overwrites outputs/ (models, tables, figures). Use git to restore the shipped results."""
import subprocess, sys, time
from pathlib import Path
here = Path(__file__).parent
quick = "--quick" in sys.argv
q = ["--quick"] if quick else []
steps = (("01_prepare.py", []), ("03_baselines.py", []), ("04_train_nn.py", q), ("05_ablation_lodo.py", q),
         ("09_final_pipeline.py", ["--skip-deploy"] if quick else []), ("10_lab_experiments.py", q), ("07_report.py", []))
t0 = time.time()
for s, extra in steps:
    print(f"\n######## {s} {' '.join(extra)}  (elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
    r = subprocess.run([sys.executable, str(here / s), *extra])
    if r.returncode:
        sys.exit(f"{s} failed")
print(f"\nALL STEPS DONE in {(time.time() - t0) / 60:.1f} min")
