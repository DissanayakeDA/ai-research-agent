"""Phase 1 check: is this machine ready to build the RAG app?

Run from the project folder, with the virtual environment activated:

    python -m scripts.check_env

It only imports packages and reads configuration. It does not download the embedding
model or call the LLM, so it is fast and free. Exit code 0 means there were no FAIL lines.
"""

import importlib
import importlib.metadata
import sys
import time

from app.config import PROJECT_ROOT, ConfigError, Settings, load_settings

# PyPI name -> the module you `import`, where they differ (default: dashes become underscores).
IMPORT_NAMES = {"python-dotenv": "dotenv"}

# Known import failures on Windows and what usually fixes them.
HINTS = {
    "torch": "Usually a missing Microsoft Visual C++ Redistributable: "
    "https://aka.ms/vs/17/release/vc_redist.x64.exe",
}


def line(status: str, message: str) -> None:
    print(f"  {status:<5} {message}")


def check_python() -> bool:
    version = ".".join(str(part) for part in sys.version_info[:3])
    ok = sys.version_info >= (3, 11)
    line("OK" if ok else "FAIL", f"Python {version}" + ("" if ok else " - this project needs 3.11+"))
    if sys.prefix == sys.base_prefix:
        line("WARN", "not inside a virtual environment - activate .venv first")
    else:
        line("OK", f"virtual environment: {sys.prefix}")
    return ok


def read_pins() -> dict[str, str]:
    """Parse the 'name==version' lines of requirements.txt, ignoring comments."""
    pins = {}
    for raw in (PROJECT_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines():
        requirement = raw.split("#", 1)[0].strip()
        if "==" in requirement:
            name, version = requirement.split("==", 1)
            pins[name.strip()] = version.strip()
    return pins


def check_packages() -> bool:
    ok = True
    for name, pinned in read_pins().items():
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            line("FAIL", f"{name} is not installed - run: pip install -r requirements.txt")
            ok = False
            continue

        # Actually importing catches broken installs, e.g. a native DLL that fails to load.
        # The time only counts modules not already loaded by an earlier import.
        module = IMPORT_NAMES.get(name, name.replace("-", "_"))
        start = time.perf_counter()
        try:
            importlib.import_module(module)
        except Exception as exc:
            hint = f"\n        {HINTS[name]}" if name in HINTS else ""
            line("FAIL", f"{name} {installed} is installed but `import {module}` failed: {exc}{hint}")
            ok = False
            continue
        seconds = time.perf_counter() - start

        if installed == pinned:
            line("OK", f"{name} {installed}  [import {seconds:.1f}s]")
        else:
            line("WARN", f"{name} {installed}, but requirements.txt pins {pinned}")

    if "torch" in sys.modules:
        import torch

        if torch.cuda.is_available():
            device = f"CUDA GPU ({torch.cuda.get_device_name(0)})"
        else:
            device = "CPU only - fine for a small embedding model"
        line("INFO", f"torch device: {device}")
    return ok


def check_config() -> Settings | None:
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        line("OK", f".env found: {env_file}")
    else:
        line("WARN", ".env not found, using defaults - create it with: Copy-Item .env.example .env")

    try:
        settings = load_settings()
    except ConfigError as exc:
        line("FAIL", str(exc))
        return None

    line("OK", f"chunk_size={settings.chunk_size} chars, chunk_overlap={settings.chunk_overlap}, top_k={settings.top_k}")
    line("OK", f"embedding model: {settings.embedding_model} (downloaded on first use, Phase 3)")
    return settings


def check_storage(settings: Settings) -> bool:
    try:
        settings.uploads_dir.mkdir(parents=True, exist_ok=True)
        settings.chroma_dir.mkdir(parents=True, exist_ok=True)
        probe = settings.data_dir / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
    except OSError as exc:
        line("FAIL", f"cannot write to {settings.data_dir}: {exc}")
        return False
    line("OK", f"data folder is writable: {settings.data_dir}")
    return True


def describe_llm(settings: Settings) -> None:
    line("INFO", f"model '{settings.llm_model}' at {settings.llm_base_url}")
    if settings.llm_is_local:
        line("INFO", "local server: no API key, no per-token charges, documents stay on this machine")
        line("INFO", "needed from Phase 5: install Ollama, then `ollama pull <model>`")
        return

    line("INFO", "hosted provider: requests may be billed (check its pricing / free tier), and")
    line("INFO", "retrieved passages from your documents are sent to it with each question")
    if settings.llm_api_key.get_secret_value():
        line("OK", "LLM_API_KEY is set")  # never print the key itself
    else:
        line("WARN", "LLM_API_KEY is empty - a hosted provider will reject requests")


def main() -> int:
    print("Python")
    ok = check_python()

    print("\nPackages (versions pinned in requirements.txt)")
    ok &= check_packages()

    print("\nConfiguration")
    settings = check_config()
    ok &= settings is not None

    if settings is not None:
        print("\nStorage")
        ok &= check_storage(settings)
        print("\nLLM (not called by this check)")
        describe_llm(settings)

    print("\nResult:", "environment ready" if ok else "fix the FAIL lines above, then run this again")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
