import argparse
import json
import tomllib
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("core", type=Path)
    parser.add_argument("binding", type=Path)
    options = parser.parse_args()
    core = options.core.resolve()
    binding = options.binding.resolve()
    manifest = binding / "Cargo.toml"
    content = manifest.read_text()
    if tomllib.loads(content).get("patch", {}).get("crates-io"):
        raise ValueError("Use a binding checkout without existing crate patches")
    lock = tomllib.loads((binding / "Cargo.lock").read_text())
    required = {
        package["name"] for package in [*lock["package"], *lock.get("patch", {}).get("unused", [])]
    }
    entries = {}
    for path in (core / "datafusion").rglob("Cargo.toml"):
        package = tomllib.loads(path.read_text()).get("package", {})
        name = package.get("name")
        if name in required and name.startswith("datafusion"):
            entries[name] = path.parent
    if "datafusion" not in entries or "datafusion-expr" not in entries:
        raise ValueError("The core checkout does not contain the required crates")
    overrides = "\n".join(
        f"{name} = {{ path = {json.dumps(str(path))} }}" for name, path in sorted(entries.items())
    )
    section = "[patch.crates-io]"
    if section in content:
        content = content.replace(section, section + "\n" + overrides, 1)
    else:
        content = content.rstrip() + "\n\n" + section + "\n" + overrides + "\n"
    manifest.write_text(content)


if __name__ == "__main__":
    main()
