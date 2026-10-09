"""Compare preserved function bodies against the pre-refactor Git snapshot.

Normalize context names, moved calls and unchanged helper extraction; retain
numeric expressions, control flow and decorators. Explicit changes are reviewed
in docs/01_工程架构报告.md. This is supporting evidence, not a proof of training.
"""

from pathlib import Path
import ast, copy, json, subprocess

root = Path(__file__).resolve().parents[1]
out = root / "verification/refactor"


def original(p):
    return subprocess.check_output(
        ["git", "show", "baseline/original:" + p], cwd=root, text=True
    )


paths = subprocess.check_output(
    ["git", "ls-tree", "-r", "--name-only", "baseline/original", "src"],
    cwd=root,
    text=True,
).splitlines()


def functions(s):
    tree = ast.parse(s)
    result = {}
    for n in tree.body:
        if isinstance(n, ast.FunctionDef):
            result[n.name] = n
        elif isinstance(n, ast.ClassDef):
            for m in n.body:
                if isinstance(m, ast.FunctionDef):
                    result[n.name + "." + m.name] = m
    return result


before = {p: functions(original(p)) for p in paths if p.endswith(".py")}
after = {
    str(p.relative_to(root)): functions(p.read_text())
    for p in (root / "src").rglob("*.py")
}
module_moves = json.loads(
    (root / "verification/refactor/module_moves.json").read_text()
)
method_modules = {}
for path, fs in after.items():
    if path.endswith(("training/experiment.py", "runtime/commands.py")):
        continue
    for name in fs:
        if "." not in name:
            method_modules[name] = path
extracts = set(before["src/ccrr/training.py"])
extracts = {s.split(".")[-1] for s in extracts if s.startswith("Experiment.")} - {
    "__init__",
    "preflight",
    "batch",
    "log",
    "mark",
    "bind_snapshot",
    "run",
}
extracts |= {
    "run",
    "source_matching",
    "reconstruction_scores",
    "public_side_features",
    "initialize_base",
}
helpers = {
    k: v
    for fs in after.values()
    for k, v in fs.items()
    if k.startswith(
        (
            "_prepare_joint",
            "_accumulate_",
            "_finalize_joint",
            "_receiver_threshold",
            "_probe_entries",
            "_fit_heads",
            "_refresh_policies",
        )
    )
}


class Normal(ast.NodeTransformer):
    def visit_Import(self, n):
        return None

    def visit_ImportFrom(self, n):
        return None

    def visit_Expr(self, n):
        if isinstance(n.value, ast.Constant) and isinstance(n.value.value, str):
            return None
        if (
            isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id in helpers
        ):
            return self.inline(n.value.func.id)
        return self.generic_visit(n)

    def visit_Assign(self, n):
        if (
            isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id in helpers
        ):
            return self.inline(n.value.func.id)
        return self.generic_visit(n)

    def visit_Return(self, n):
        if (
            isinstance(n.value, ast.Call)
            and isinstance(n.value.func, ast.Name)
            and n.value.func.id in helpers
        ):
            return self.inline(n.value.func.id)
        return self.generic_visit(n)

    def inline(self, name):
        b = copy.deepcopy(helpers[name].body)
        if name != "_finalize_joint" and isinstance(b[-1], ast.Return):
            b = b[:-1]
        module = ast.Module(body=b, type_ignores=[])
        return self.visit(module).body

    def visit_Name(self, n):
        if n.id in {"experiment", "study"}:
            n.id = "self"
        if n.id == "CounterLocal":
            n.id = "Counter"
        return n

    def visit_arg(self, n):
        if n.arg in {"experiment", "study"}:
            n.arg = "self"
        return n

    def visit_Call(self, n):
        if isinstance(n.func, ast.Name) and n.func.id == "quantize_latent":
            return self.visit(
                ast.Call(
                    func=ast.Name(id="ste_round", ctx=ast.Load()),
                    args=[ast.BinOp(left=n.args[0], op=ast.Div(), right=n.args[1])],
                    keywords=[],
                )
            )
        if isinstance(n.func, ast.Name) and n.func.id == "dequantize_latent":
            return self.visit(ast.BinOp(left=n.args[0], op=ast.Mult(), right=n.args[1]))
        method = (
            n.func.attr
            if isinstance(n.func, ast.Attribute)
            else n.func.id
            if isinstance(n.func, ast.Name)
            else None
        )
        if method in extracts:
            receiver = n.func.value if isinstance(n.func, ast.Attribute) else None
            if (isinstance(receiver, ast.Name) and receiver.id in {"self", "e"}) or (
                isinstance(receiver, ast.Attribute)
                and ast.unparse(receiver).startswith("self.")
            ):
                n.args.insert(0, receiver)
            n.func = ast.Name(id=method, ctx=ast.Load())
        # Avoid visit_Attribute rewriting function-bound calls; args normalize normally.
        return self.generic_visit(n)

    def visit_Attribute(self, n):
        # Original dispatch bound method and its replacement lambda have equal normalization.
        if (
            isinstance(n.value, ast.Name)
            and n.value.id == "self"
            and n.attr in extracts
        ):
            return ast.Lambda(
                args=ast.arguments(
                    posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]
                ),
                body=ast.Call(
                    func=ast.Name(id=n.attr, ctx=ast.Load()),
                    args=[ast.Name(id="self", ctx=ast.Load())],
                    keywords=[],
                ),
            )
        return self.generic_visit(n)


def normalize(node):
    n = Normal().visit(copy.deepcopy(node))
    return ast.dump(n, include_attributes=False)


# Map moved classes by class-qualified function name. Only use uniqueness.
index = {}
for p, fs in after.items():
    for name, n in fs.items():
        index.setdefault(name, []).append((p, n))
rows = []
for p, fs in before.items():
    module = p.removeprefix("src/").removesuffix(".py").replace("/", ".")
    mapped = "src/" + module_moves.get(module, module).replace(".", "/") + ".py"
    for name, n in fs.items():
        if (
            p == "src/ccrr/training.py"
            and name.startswith("Experiment.")
            and name.split(".")[-1] in extracts
        ):
            targetname = name.split(".")[-1]
            target = (
                "src/ccrr/training/runner.py"
                if targetname == "run"
                else method_modules.get(targetname)
            )
        elif p == "src/ccrr/final_study.py" and name.split(".")[-1] in {
            "source_matching",
            "reconstruction_scores",
            "public_side_features",
        }:
            targetname = name.split(".")[-1]
            target = "src/ccrr/evaluation/attack_scoring.py"
        elif name == "load_system" and p == "src/ccrr/cli.py":
            targetname = name
            target = "src/ccrr/runtime/commands.py"
        else:
            targetname = name
            target = mapped
        candidates = index.get(targetname, [])
        pair = next(((q, v) for q, v in candidates if q == target), None)
        if pair is None and len(candidates) == 1:
            pair = candidates[0]
        if pair is None:
            rows.append(
                {
                    "before": p + ":" + name,
                    "status": "explicit_exception",
                    "reason": "CLI main redesigned as dispatch"
                    if name == "main"
                    else "Removed trivial CounterLocal wrapper",
                }
            )
            continue
        q, v = pair
        old = copy.deepcopy(n)
        old.name = v.name
        equal = normalize(old) == normalize(v)
        rows.append(
            {
                "before": p + ":" + name,
                "after": q + ":" + v.name,
                "status": "normalized_identical" if equal else "review_difference",
            }
        )
        if not equal:
            d = out / "ast_diffs"
            d.mkdir(exist_ok=True)
            (d / (name.replace(".", "_") + ".old")).write_text(
                ast.unparse(Normal().visit(old))
            )
            (d / (name.replace(".", "_") + ".new")).write_text(
                ast.unparse(Normal().visit(copy.deepcopy(v)))
            )


def statements_equal(left, right):
    return ast.dump(
        Normal().visit(ast.Module(body=copy.deepcopy(left), type_ignores=[])),
        include_attributes=False,
    ) == ast.dump(
        Normal().visit(ast.Module(body=copy.deepcopy(right), type_ignores=[])),
        include_attributes=False,
    )


old_main = before["src/ccrr/cli.py"]["main"]
new_main = after["src/ccrr/cli.py"]["main"]
old_try = next(n for n in old_main.body if isinstance(n, ast.Try))
new_try = next(n for n in new_main.body if isinstance(n, ast.Try))
cli = {
    "parser_unchanged": statements_equal(
        old_main.body[: old_main.body.index(old_try)],
        new_main.body[: new_main.body.index(new_try)],
    ),
    "error_handler_unchanged": statements_equal(old_try.handlers, new_try.handlers),
    "handlers": {},
}
for branch in old_try.body:
    while isinstance(branch, ast.If):
        command = branch.test.comparators[0].value
        handler = after["src/ccrr/runtime/commands.py"][command.replace("-", "_")]
        body = handler.body
        # Common original main() returned zero after these branch bodies.
        if (
            body
            and isinstance(body[-1], ast.Return)
            and isinstance(body[-1].value, ast.Constant)
            and body[-1].value.value == 0
        ):
            body = body[:-1]
        cli["handlers"][command] = statements_equal(branch.body, body)
        branch = branch.orelse[0] if branch.orelse else None

allowed_review = {
    "src/ccrr/cli.py:main",
    "src/ccrr/training.py:Experiment.__init__",
    "src/ccrr/training.py:Experiment.preflight",
}
allowed_removed = {"src/ccrr/training.py:CounterLocal"}
unexpected = [
    r
    for r in rows
    if (r["status"] == "review_difference" and r["before"] not in allowed_review)
    or (r["status"] == "explicit_exception" and r["before"] not in allowed_removed)
]
report = {
    "scope": "AST bodies normalized only for imports, documented function moves/context renaming, helper extraction and unchanged quantizer wrappers; not an execution proof",
    "rows": rows,
    "counts": {
        s: sum(r["status"] == s for r in rows) for s in {r["status"] for r in rows}
    },
    "decorators_compared": True,
    "cli": cli,
    "unexpected_differences": unexpected,
}
(out / "function_body_equivalence.json").write_text(
    json.dumps(report, ensure_ascii=False, indent=2) + "\n"
)
print(report["counts"])
for r in rows:
    if r["status"] != "normalized_identical":
        print(r)
print("CLI equivalence:", cli)
if (
    unexpected
    or not cli["parser_unchanged"]
    or not cli["error_handler_unchanged"]
    or not all(cli["handlers"].values())
):
    raise SystemExit("Unexpected refactor difference; review before accepting")
