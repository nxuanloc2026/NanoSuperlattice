"""Print the installed BornAgain and Dans_Diffraction APIs the adapters rely on.

    python -m integrations.probe > probe.txt

SWIG-wrapped BornAgain classes list their C++ constructor prototypes when
called with an impossible argument, which is what the adapter needs to know.
"""

import importlib
import importlib.metadata
import inspect
import pkgutil
import re


class _Impossible:
    pass


def _prototypes(obj):
    try:
        obj(_Impossible())
    except TypeError as exc:
        return str(exc).strip()
    except Exception as exc:  # noqa: BLE001 - report anything the probe hits
        return f"{type(exc).__name__}: {exc}"
    return "accepted an arbitrary object (no prototype list)"


def _methods(cls):
    return ", ".join(m for m in dir(cls) if not m.startswith("_"))


def _version(dist):
    try:
        return importlib.metadata.version(dist)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def bornagain():
    try:
        import bornagain as ba
    except ImportError as exc:
        print("BornAgain not importable:", exc)
        return
    print("BornAgain", getattr(ba, "__version__", None) or _version("bornagain"))
    public = sorted(n for n in dir(ba) if not n.startswith("_"))
    print("\nALL top-level names:\n" + ", ".join(public))

    # Classes we need to locate after renames in BornAgain 25.
    pattern = re.compile(r"Layout|Detector|Color|Colour|Datafield|Interference|Layer$|Material", re.I)
    for name in public:
        obj = getattr(ba, name)
        if pattern.search(name) and callable(obj):
            print(f"\n=== {name}")
            print("constructors:", _prototypes(obj))
            if inspect.isclass(obj):
                print("methods:", _methods(obj))

    print("\n=== MaterialColor in submodules")
    found = False
    for info in pkgutil.walk_packages(ba.__path__, "bornagain."):
        try:
            module = importlib.import_module(info.name)
        except Exception:  # noqa: BLE001
            continue
        for attr in dir(module):
            if "color" in attr.lower() or "colour" in attr.lower():
                obj = getattr(module, attr)
                found = True
                print(f"{info.name}.{attr}:", _prototypes(obj) if callable(obj) else repr(obj))
    if not found:
        print("none found")

    for name in ("Particle", "Compound", "RotationEuler", "Sample", "Layer"):
        cls = getattr(ba, name, None)
        if cls is not None:
            print(f"\n{name} methods:", _methods(cls))


def dans():
    try:
        import Dans_Diffraction as dif
    except ImportError as exc:
        print("Dans_Diffraction not importable:", exc)
        return
    print("\n\nDans_Diffraction", getattr(dif, "__version__", None) or _version("Dans_Diffraction"))
    try:
        scatter = type(dif.Crystal().Scatter)
    except Exception as exc:  # noqa: BLE001
        print("could not create an empty Crystal:", exc)
        return
    for name in ("setup_scatter", "intensity"):
        print(f"{scatter.__name__}.{name}{inspect.signature(getattr(scatter, name))}")


if __name__ == "__main__":
    bornagain()
    dans()
