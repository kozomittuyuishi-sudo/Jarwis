"""
AG Live Verification Script
Exercises the AG pipeline non-interactively using direct API calls.
"""
import sys
import os
import traceback

# Ensure D:\AG (project root) is on sys.path regardless of where we run from
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
os.chdir(_ROOT)  # also chdir so relative imports (dotenv etc.) resolve

PASS = []
FAIL = []

def check(label, condition, detail=""):
    if condition:
        PASS.append(label)
        print(f"  [PASS] {label}")
    else:
        FAIL.append(label)
        print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))

# ─────────────────────────────────────────────────────────
# 1. PreprocessorBrain — self-info queries
# ─────────────────────────────────────────────────────────
print("\n[1] PreprocessorBrain — self-info queries")
try:
    from preprocessor_brain import PreprocessorBrain

    pb = PreprocessorBrain()

    r = pb.process("what can you do")
    check("can_you_do -> handled locally",     not r.requires_llm)
    check("can_you_do -> has result",          r.result is not None or r.error is not None)

    r = pb.process("what are your limitations")
    check("limitations -> handled locally",    not r.requires_llm)
    check("limitations -> has result",         r.result is not None or r.error is not None)

    r = pb.process("what brain are you using")
    check("brain_info -> returns a result", r is not None and not (r.requires_llm is None))

    r = pb.process("tell me about the weather today")
    check("weather -> requires LLM",           r.requires_llm)

    r = pb.process("")
    check("empty input -> does not crash",     r is not None)
    check("empty input -> has error or result or requires_llm", r.requires_llm or r.result is not None or r.error is not None)

except Exception as e:
    print(f"  [ERROR] PreprocessorBrain block failed: {e}")
    traceback.print_exc()
    FAIL.append("PreprocessorBrain block")

# ─────────────────────────────────────────────────────────
# 2. Intent router
# ─────────────────────────────────────────────────────────
print("\n[2] Intent router")
try:
    from pipeline.intent_router import classify, Intent

    result = classify("list all files")
    check("list_files -> not unknown",        result.intent.value != "UNKNOWN")

    result = classify("who are you")
    check("who_are_you -> has intent",        result.intent is not None)

    result = classify("read README.md")
    check("read_file -> has intent",          result.intent is not None)

except Exception as e:
    print(f"  [ERROR] Intent router block failed: {e}")
    traceback.print_exc()
    FAIL.append("Intent router block")

# ─────────────────────────────────────────────────────────
# 3. FallbackClassifier
# ─────────────────────────────────────────────────────────
print("\n[3] FallbackClassifier")
try:
    from pipeline.fallback_classifier import get_fallback_classifier, classify_failure

    fc = get_fallback_classifier()
    check("fallback_classifier instantiates", fc is not None)

    result = classify_failure("Connection refused")
    check("classify_failure -> returns result", result is not None)
    check("classify_failure -> has message attr", hasattr(result, "message") or hasattr(result, "user_message") or isinstance(result, str))

except Exception as e:
    print(f"  [ERROR] FallbackClassifier block failed: {e}")
    traceback.print_exc()
    FAIL.append("FallbackClassifier block")

# ─────────────────────────────────────────────────────────
# 4. DirectoryControl
# ─────────────────────────────────────────────────────────
print("\n[4] DirectoryControl")
try:
    from directory_control import DirectoryControl
    import tempfile, pathlib

    dc = DirectoryControl()
    check("DirectoryControl instantiates", dc is not None)
    check("no active workspace initially", dc.get_active_directory() is None)

    with tempfile.TemporaryDirectory() as tmp:
        dc.set_directory(tmp)
        check("set_directory works",        dc.get_active_directory() is not None)

        entries = dc.list_directory()
        check("list_directory returns list", isinstance(entries, list))

        # create and read a file
        dc.create_file("hello.txt", "Hello, AG!")
        content = dc.read_file("hello.txt")
        check("create + read file works",   "Hello, AG!" in content)

        dc.clear_directory()
        check("clear_directory works",      dc.get_active_directory() is None)

except Exception as e:
    print(f"  [ERROR] DirectoryControl block failed: {e}")
    traceback.print_exc()
    FAIL.append("DirectoryControl block")

# ─────────────────────────────────────────────────────────
# 5. WorkingMemory
# ─────────────────────────────────────────────────────────
print("\n[5] WorkingMemory")
try:
    from memory.working_memory import WorkingMemory

    wm = WorkingMemory()
    wm.set_objective("Run verification")
    check("set_objective works",            wm.get_objective() == "Run verification")

    wm.add_fact("test_key", "test_value")
    check("add_fact / get_fact works",      wm.get_fact("test_key") == "test_value")

    ctx = wm.build_context()
    check("build_context returns dict",     isinstance(ctx, dict))

except Exception as e:
    print(f"  [ERROR] WorkingMemory block failed: {e}")
    traceback.print_exc()
    FAIL.append("WorkingMemory block")

# ─────────────────────────────────────────────────────────
# 6. Jarwis startup — import Jarwis.py directly
# ─────────────────────────────────────────────────────────
print("\n[6] AG module import")
try:
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location("Jarwis", "Jarwis.py")
    ag_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ag_module)
    check("Jarwis.py loads without error", True)
    check("Jarwis.py has main()",          hasattr(ag_module, "main"))
    check("Jarwis.py has startup()",       hasattr(ag_module, "startup"))
except Exception as e:
    print(f"  [ERROR] Jarwis.py import failed: {e}")
    traceback.print_exc()
    FAIL.append("Jarwis.py import")

# ─────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"PASSED: {len(PASS)}   FAILED: {len(FAIL)}")
if FAIL:
    print("Failed checks:")
    for f in FAIL:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All verification checks passed.")
    sys.exit(0)
