"""Inspect current source architecture, optionally comparing a Git baseline.

Static imports include lazy imports. Calls are syntax-level evidence, not a
claim to resolve dynamic module types or prove numerical behavior.
"""

import argparse
import ast
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True)


def module_name(path):
    name = path.removeprefix("src/").removesuffix(".py").replace("/", ".")
    return name.removesuffix(".__init__")


def components(graph):
    """Tarjan SCCs. A one-module component is a cycle only with a self edge."""
    index, low, stack, active, result = {}, {}, [], set(), []

    def visit(node):
        index[node] = low[node] = len(index)
        stack.append(node)
        active.add(node)
        for target in sorted(graph[node]):
            if target not in index:
                visit(target)
                low[node] = min(low[node], low[target])
            elif target in active:
                low[node] = min(low[node], index[target])
        if low[node] == index[node]:
            group = []
            while True:
                item = stack.pop()
                active.remove(item)
                group.append(item)
                if item == node:
                    break
            if len(group) > 1 or node in graph[node]:
                result.append(sorted(group))

    for node in sorted(graph):
        if node not in index:
            visit(node)
    return result


def analyze(sources):
    modules = {module_name(p): (p, s) for p, s in sources.items()}
    graph, files, bodies = {}, {}, {}
    for name, (path, source) in modules.items():
        tree = ast.parse(source)
        dependencies = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                targets = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                parent = (
                    name if path.endswith("/__init__.py") else name.rsplit(".", 1)[0]
                )
                if node.level:
                    parts = parent.split(".")[: len(parent.split(".")) - node.level + 1]
                    base = ".".join(parts + ([node.module] if node.module else []))
                else:
                    base = node.module or ""
                targets = [base] + [base + "." + alias.name for alias in node.names]
            else:
                continue
            dependencies.update(t for t in targets if t in modules and t != name)
        graph[name] = dependencies
        definitions = []

        def definitions_in(nodes, prefix=""):
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    definitions_in(node.body, prefix + node.name + ".")
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    qualified = name + "." + prefix + node.name
                    body = ast.dump(
                        ast.Module(body=node.body, type_ignores=[]),
                        include_attributes=False,
                    )
                    digest = hashlib.sha256(body.encode()).hexdigest()
                    bodies.setdefault(digest, []).append(qualified)
                    definitions.append(
                        {
                            "symbol": prefix + node.name,
                            "line": node.lineno,
                            "lines": node.end_lineno - node.lineno + 1,
                            "calls": sorted(
                                set(
                                    ast.unparse(n.func)
                                    for n in ast.walk(node)
                                    if isinstance(n, ast.Call)
                                )
                            ),
                        }
                    )

        definitions_in(tree.body)
        files[path] = {
            "module": name,
            "lines": len(source.splitlines()),
            "sha256": hashlib.sha256(source.encode()).hexdigest(),
            "imports": sorted(dependencies),
            "definitions": definitions,
        }
    return {
        "files": files,
        "python_files": len(files),
        "source_lines": sum(f["lines"] for f in files.values()),
        "import_cycles_including_lazy": components(graph),
        "identical_function_bodies": [v for v in bodies.values() if len(v) > 1],
        "longest_functions": sorted(
            [
                dict(file=path, **definition)
                for path, f in files.items()
                for definition in f["definitions"]
            ],
            key=lambda d: d["lines"],
            reverse=True,
        )[:20],
        "packages": dict(
            Counter(name.split(".")[1] if "." in name else name for name in modules)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", help="Optional existing Git commit or ref")
    parser.add_argument("--out", required=True, help="Output JSON path")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    after = {
        str(p.relative_to(root)): p.read_text()
        for p in sorted((root / "src").rglob("*.py"))
    }
    report = {
        "baseline_commit": None,
        "comparison_performed": False,
        "scope": "Static source inventory; lazy imports included; dynamic calls not resolved",
        "after": analyze(after),
    }
    if args.baseline:
        baseline = git(
            root, "rev-parse", "--verify", args.baseline + "^{commit}"
        ).strip()
        paths = git(root, "ls-tree", "-r", "--name-only", baseline, "src").splitlines()
        before = {
            p: git(root, "show", baseline + ":" + p) for p in paths if p.endswith(".py")
        }
        report.update(
            baseline_commit=baseline,
            comparison_performed=True,
            before=analyze(before),
        )
    output = root / args.out
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(
        json.dumps(
            {
                phase: {
                    key: report[phase][key]
                    for key in (
                        "python_files",
                        "source_lines",
                        "import_cycles_including_lazy",
                    )
                }
                for phase in ("before", "after")
                if phase in report
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
