"""Pre-flight checks: Ollama reachability, model availability, data sanity."""

from __future__ import annotations

from typing import List, Tuple

import ollama


def check_ollama_reachable(settings) -> Tuple[bool, str]:
    """Return (ok, message) for Ollama connectivity."""
    try:
        client = ollama.Client(host=settings.ollama_host)
        client.list()
        return True, f"Ollama reachable at {settings.ollama_host}"
    except Exception as e:
        return False, f"Cannot reach Ollama at {settings.ollama_host}: {e}"


def check_models_available(settings) -> List[Tuple[bool, str]]:
    """Check each configured model is pulled. Returns list of (ok, message)."""
    results = []
    try:
        client = ollama.Client(host=settings.ollama_host)
        pulled = {m.model for m in client.list().models}
    except Exception as e:
        return [(False, f"Cannot list models: {e}")]

    for model in settings.models:
        if model in pulled:
            results.append((True, f"  {model} — available"))
        else:
            results.append((False, f"  {model} — NOT FOUND (run: ollama pull {model})"))
    return results


def check_model_digests(settings) -> List[Tuple[str, str]]:
    """Return (model, digest) pairs for reproducibility logging."""
    pairs = []
    try:
        client = ollama.Client(host=settings.ollama_host)
        for model in settings.models:
            try:
                info = client.show(model)
                digest = getattr(info, "digest", None) or "unknown"
                pairs.append((model, digest))
            except Exception:
                pairs.append((model, "unavailable"))
    except Exception:
        pass
    return pairs


def check_input_data(settings) -> Tuple[bool, str]:
    """Verify input file and webdata directory."""
    if not settings.input_file.exists():
        return False, f"Input file missing: {settings.input_file}"
    with open(settings.input_file) as f:
        urls = [line.strip() for line in f if line.strip().startswith("http")]
    if not urls:
        return False, f"No URLs found in {settings.input_file}"
    n_articles = sum(
        len(list(src_dir.glob("*.json")))
        for src_dir in settings.all_source_webdata_dirs
    )
    return True, f"{len(urls)} URLs in input file, {n_articles} scraped articles on disk"


def validate(settings) -> bool:
    """Run all pre-flight checks. Returns True if all pass."""
    all_ok = True

    print("\n--- Validation ---")

    client = None
    pulled: set[str] = set()
    try:
        client = ollama.Client(host=settings.ollama_host)
        model_list = client.list()
        pulled = {m.model for m in model_list.models}
        ok = True
        msg = f"Ollama reachable at {settings.ollama_host}"
    except Exception as e:
        ok = False
        msg = f"Cannot reach Ollama at {settings.ollama_host}: {e}"
    print(f"{'[ok]' if ok else '[FAIL]'} {msg}")
    if not ok:
        all_ok = False

    print("\nModel availability:")
    if pulled:
        for model in settings.models:
            model_ok = model in pulled
            msg = (
                f"  {model} — available"
                if model_ok
                else f"  {model} — NOT FOUND (run: ollama pull {model})"
            )
            print(f"{'[ok]' if model_ok else '[FAIL]'} {msg}")
            if not model_ok:
                all_ok = False
    else:
        print("[FAIL] Cannot list models because Ollama is unreachable.")
        all_ok = False

    print("\nModel digests (for reproducibility):")
    if client is None:
        for model in settings.models:
            print(f"  {model}: unavailable")
    else:
        for model in settings.models:
            try:
                info = client.show(model)
                digest = getattr(info, "digest", None) or "unknown"
            except Exception:
                digest = "unavailable"
            print(f"  {model}: {digest}")

    ok, msg = check_input_data(settings)
    print(f"\n{'[ok]' if ok else '[FAIL]'} {msg}")
    if not ok:
        all_ok = False

    print(f"\nPrompt hash: {settings.prompt_hash}")
    print(f"Max workers: {settings.max_workers}")
    print(f"Max article chars: {settings.max_article_chars}")

    if all_ok:
        print("\nAll checks passed.")
    else:
        print("\nSome checks FAILED — fix issues above before running the pipeline.")

    return all_ok
