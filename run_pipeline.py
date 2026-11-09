# Run the full news-regime discovery pipeline end to end in Colab.
# All notebook output paths are preserved under /content/drive/MyDrive/TTIC31220-Spring26/newsheadline.

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import runpy
import subprocess
import sys
import time
import traceback
from pathlib import Path

try:
    from google.colab import drive  # type: ignore
    if not os.path.exists("/content/drive/MyDrive"):
        drive.mount("/content/drive")
except Exception:
    pass

PREFERRED_PROJECT_DIR = Path("/content/drive/MyDrive/TTIC31220-Spring26/newsheadline")
PROJECT_DIR = PREFERRED_PROJECT_DIR if PREFERRED_PROJECT_DIR.exists() else Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "_data"
REQUIREMENTS_FILE = PROJECT_DIR / "requirements-project.txt"

STEPS = [
    ("01_market_data.py", "Market features"),
    ("02_news_features.py", "RavenPack DJPR daily news features"),
    ("02d_news_macro.py", "WRDS macro news features (optional)"),
    ("03_embedding_colab.py", "FinBERT embeddings"),
    ("04_topic_models.py", "LDA + BERTopic"),
    ("05_baseline_price.py", "Price-only baselines"),
    ("06_text_hybrid_models.py", "Text / hybrid / alt regime models"),
    ("07_evaluation.py", "Evaluation summary"),
    ("08_regime_characterization.py", "Final regime characterization assets"),
]

# Primary output per step (used by --skip-existing).
KEY_OUTPUTS: dict[str, Path] = {
    "01_market_data.py": DATA_DIR / "market_features.parquet",
    "02_news_features.py": DATA_DIR / "news_features_daily.parquet",
    "02d_news_macro.py": DATA_DIR / "news_features_macro_daily.parquet",
    "03_embedding_colab.py": DATA_DIR / "daily_finbert_embeddings.npz",
    "04_topic_models.py": DATA_DIR / "lda_topics_k15_daily.parquet",
    "05_baseline_price.py": DATA_DIR / "regimes_price.parquet",
    "06_text_hybrid_models.py": DATA_DIR / "regimes_text_scalar.parquet",
    "07_evaluation.py": DATA_DIR / "eval_summary.parquet",
}

# Extra outputs checked for multi-artifact steps.
EXTRA_OUTPUTS: dict[str, list[Path]] = {
    "06_text_hybrid_models.py": [
        DATA_DIR / "regimes_lean_hybrid.parquet",
        DATA_DIR / "regimes_text_btopic.parquet",
    ],
}

OPTIONAL_FLAG_STEPS = {
    "03_embedding_colab.py",
    "04_topic_models.py",
    "06_text_hybrid_models.py",
    "02d_news_macro.py",
}

# Steps that should only run when their dedicated flag is set
# (otherwise quietly skipped).  Reserved for future use; currently none.
FLAG_GATED_STEPS: dict[str, str] = {}

# Steps that bypass --skip-existing because they have their own internal
# incremental cache and are cheap to invoke on unchanged inputs.  Re-running
# them is harmless (they short-circuit per-pipeline) and ensures the master
# summary reflects the latest regime labels without manual file deletion.
ALWAYS_RUN_STEPS: set[str] = {"07_evaluation.py"}


def optional_outputs(enable_macro: bool = False) -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    if enable_macro:
        for script, path in [
            ("02d_news_macro.py", DATA_DIR / "news_features_macro_daily.parquet"),
            ("03_embedding_colab.py", DATA_DIR / "daily_finbert_embeddings_macro.npz"),
            ("04_topic_models.py", DATA_DIR / "lda_topics_macro_k15_daily.parquet"),
            ("04_topic_models.py", DATA_DIR / "bertopic_macro_daily.parquet"),
            ("06_text_hybrid_models.py", DATA_DIR / "regimes_macro_supplement.parquet"),
            ("06_text_hybrid_models.py", DATA_DIR / "regimes_macro_text_supplement.parquet"),
        ]:
            out.setdefault(script, []).append(path)
    return out


def step_outputs_exist(script: str, *, enable_macro: bool = False) -> bool:
    primary = KEY_OUTPUTS.get(script)
    if primary is None:
        return False
    if not primary.exists():
        return False
    for extra in EXTRA_OUTPUTS.get(script, []):
        if not extra.exists():
            return False
    for opt in optional_outputs(enable_macro).get(script, []):
        if not opt.exists():
            return False
    return True


def run_step(
    script: str,
    label: str,
    *,
    subprocess_mode: bool = False,
    extra_argv: list[str] | None = None,
) -> None:
    start = time.time()
    script_path = PROJECT_DIR / script
    print(f"\n=== {script}: {label} ===", flush=True)

    argv = [str(script_path)] + (extra_argv or [])

    if subprocess_mode:
        proc = subprocess.run([sys.executable, "-u", *argv], cwd=str(PROJECT_DIR))
        elapsed = time.time() - start
        if proc.returncode != 0:
            raise SystemExit(f"Step failed: {script} (exit code {proc.returncode})")
        print(f"=== finished {script} in {elapsed / 60:.1f} min ===", flush=True)
        return

    old_cwd = Path.cwd()
    old_argv = sys.argv[:]
    try:
        os.chdir(PROJECT_DIR)
        sys.argv = argv
        runpy.run_path(str(script_path), run_name="__main__")
    except SystemExit as exc:
        code = exc.code if isinstance(exc.code, int) else 1
        if code != 0:
            raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(f"Step failed: {script}")
    finally:
        sys.argv = old_argv
        os.chdir(old_cwd)

    elapsed = time.time() - start
    print(f"=== finished {script} in {elapsed / 60:.1f} min ===", flush=True)


def ensure_package(import_name: str, pip_name: str | None = None) -> None:
    if importlib.util.find_spec(import_name) is not None:
        print(f"Dependency already available: {import_name}", flush=True)
        return
    package = pip_name or import_name
    print(f"Installing missing dependency: {package}", flush=True)
    proc = subprocess.run([sys.executable, "-m", "pip", "install", package])
    if proc.returncode != 0:
        raise SystemExit(f"Failed to install dependency: {package}")


def ensure_requirements(*, force: bool = False) -> None:
    if not REQUIREMENTS_FILE.exists():
        print(f"No project requirements file found at {REQUIREMENTS_FILE}; skipping dependency install.", flush=True)
        return
    req_bytes = REQUIREMENTS_FILE.read_bytes()
    req_hash = hashlib.sha256(req_bytes).hexdigest()[:12]
    marker = Path("/tmp") / f"newsheadline_requirements_{req_hash}.installed"
    if marker.exists() and not force:
        print(f"Project requirements already installed for this runtime: {marker}", flush=True)
        return
    print(f"Installing dependencies from {REQUIREMENTS_FILE}", flush=True)
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-r", str(REQUIREMENTS_FILE)],
        cwd=str(PROJECT_DIR),
    )
    if proc.returncode != 0:
        raise SystemExit(f"Failed to install dependencies from {REQUIREMENTS_FILE}")
    marker.write_text(time.strftime("%Y-%m-%d %H:%M:%S"), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete unsupervised news-regime pipeline in Colab.")
    parser.add_argument("--start-at", choices=[s for s, _ in STEPS], help="Start from this script instead of 01.")
    parser.add_argument("--stop-after", choices=[s for s, _ in STEPS], help="Stop after this script.")
    parser.add_argument(
        "--enable-macro",
        action="store_true",
        default=True,
        help="Enable WRDS macro pull (02d) and supplement models (06). "
             "Default: enabled.  Pass --no-macro to disable.",
    )
    parser.add_argument(
        "--no-macro",
        dest="enable_macro",
        action="store_false",
        help="Disable WRDS macro pull (02d) and supplement models (06).",
    )
    parser.add_argument(
        "--skip-expensive-text",
        action="store_true",
        help="Skip 03_embedding_colab.py and 04_topic_models.py if their key outputs already exist.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        default=True,
        help="Skip any step whose key output file(s) already exist.  "
             "Default: enabled.  Pass --no-skip-existing to force a fresh run.",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Disable skip-existing: re-run every step regardless of cached outputs.",
    )
    parser.add_argument(
        "--subprocess",
        action="store_true",
        help="Run each step in a separate Python process. Default is in-process for notebook/WRDS prompt compatibility.",
    )
    parser.add_argument(
        "--no-install-requirements",
        action="store_true",
        help="Do not install requirements.txt before running the pipeline.",
    )
    parser.add_argument(
        "--force-install-requirements",
        action="store_true",
        help="Install requirements.txt even if this runtime was already marked as installed.",
    )
    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring notebook/kernel arguments: {unknown}", flush=True)

    if not args.no_install_requirements:
        ensure_requirements(force=args.force_install_requirements)
    ensure_package("wrds")

    flag_argv: list[str] = []
    if args.enable_macro:
        flag_argv.append("--enable-macro")

    steps = STEPS
    if args.start_at:
        start_idx = next(i for i, (script, _) in enumerate(steps) if script == args.start_at)
        steps = steps[start_idx:]
    if args.stop_after:
        stop_idx = next(i for i, (script, _) in enumerate(steps) if script == args.stop_after)
        steps = steps[: stop_idx + 1]

    for script, label in steps:
        # Skip flag-gated steps when the gating flag is off.
        gate_attr = FLAG_GATED_STEPS.get(script)
        if gate_attr is not None and not getattr(args, gate_attr, False):
            print(f"\n=== skipping {script}: --{gate_attr.replace('_', '-')} not set ===", flush=True)
            continue
        if args.skip_existing and script in KEY_OUTPUTS and script not in ALWAYS_RUN_STEPS:
            if step_outputs_exist(script, enable_macro=args.enable_macro):
                print(f"\n=== skipping {script}: existing output(s) found ===", flush=True)
                continue
        if args.skip_expensive_text and script == "03_embedding_colab.py":
            if (DATA_DIR / "daily_finbert_embeddings.npz").exists():
                print(f"\n=== skipping {script}: existing daily_finbert_embeddings.npz found ===", flush=True)
                continue
        if args.skip_expensive_text and script == "04_topic_models.py":
            if (DATA_DIR / "lda_topics_k15_daily.parquet").exists():
                print(f"\n=== skipping {script}: existing LDA topic file found ===", flush=True)
                continue

        extra = list(flag_argv) if script in OPTIONAL_FLAG_STEPS else None
        # 06 has an internal "--skip-existing-regimes" flag that skips already-
        # produced regimes_*.parquet sections.  Propagate --skip-existing from
        # the orchestrator so a partial re-run (e.g. --enable-macro on top of a
        # fully-populated _data dir) doesn't re-fit every section.
        if script == "06_text_hybrid_models.py" and args.skip_existing:
            extra = (extra or []) + ["--skip-existing-regimes"]
        run_step(script, label, subprocess_mode=args.subprocess, extra_argv=extra)


if __name__ == "__main__":
    main()
