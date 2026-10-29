import ast
from importlib.util import resolve_name
from pathlib import Path

from kernel_workflow.rsi.loop.inner import ROLES
from kernel_workflow.rsi.reverse_engineering.analyst import ROLE
from kernel_workflow.rsi.runtime import RuntimeAgent

ROOT = Path(__file__).resolve().parents[1]


def implementation_graph():
    graph = {}
    for path in ROOT.rglob("*.py"):
        relative = path.relative_to(ROOT)
        if "tests" in relative.parts:
            continue
        names = list(relative.with_suffix("").parts)
        if names[-1] == "__init__":
            names.pop()
        module = ".".join(("kernel_workflow", "rsi", *names))
        package = module if path.name == "__init__.py" else module.rpartition(".")[0]
        dependencies = set()
        for item in ast.walk(ast.parse(path.read_text())):
            if isinstance(item, ast.ImportFrom):
                dependencies.add(resolve_name("." * item.level + (item.module or ""), package))
            elif isinstance(item, ast.Import):
                dependencies.update(value.name for value in item.names)
        graph[module] = dependencies
    return graph


def test_three_packages_own_their_mechanisms_and_retire_flat_modules():
    expected = {
        "loop": ("outer", "inner", "online", "evaluator", "qualification", "run"),
        "reverse_engineering": ("model", "trace", "investigate", "analyst", "sources/archive", "sources/archive_tree",
                                "sources/capture", "sources/kernelwiki"),
        "replay_buffer": ("build", "store", "cohorts", "simulator", "objective", "evaluate"),
    }
    for package, modules in expected.items():
        assert (ROOT / package / "tests").is_dir()
        assert all((ROOT / package / (module + ".py")).is_file() for module in modules)
    retired = ("audit", "reverse", "buffer", "evaluate", "objective", "replay", "evolve", "online",
               "history", "policy", "rollout", "capture", "loop")
    assert not any((ROOT / (name + ".py")).exists() for name in retired)


def test_reverse_and_runtime_do_not_depend_on_loop_or_replay():
    for module, dependencies in implementation_graph().items():
        if module.startswith("kernel_workflow.rsi.reverse_engineering"):
            assert not any(value.startswith(("kernel_workflow.rsi.loop", "kernel_workflow.rsi.replay_buffer"))
                           for value in dependencies), module
        if module == "kernel_workflow.rsi.runtime":
            assert not any(value.startswith(("kernel_workflow.rsi.loop", "kernel_workflow.rsi.reverse_engineering",
                                              "kernel_workflow.rsi.replay_buffer")) for value in dependencies)


def test_implementation_imports_have_no_architectural_cycle():
    graph = implementation_graph()
    visited, active = set(), set()

    def visit(module):
        assert module not in active, module
        if module in visited:
            return
        active.add(module)
        for dependency in graph[module] & graph.keys():
            visit(dependency)
        active.remove(module)
        visited.add(module)

    for module in graph:
        visit(module)


def test_agent_schemas_are_owned_by_inner_loop_and_method_analyst():
    assert set(ROLES) == {"preflight", "developer", "reviewer"}
    assert ROLES["developer"].instructions.parent.resolve() == ROOT / "loop/roles"
    assert ROLE.instructions.resolve() == ROOT / "reverse_engineering/roles/analyst.md"
    assert ROLE.schema["properties"]["method_status"]["enum"][-1] == "unknown"
    assert RuntimeAgent.__module__ == "kernel_workflow.rsi.runtime"


def test_trace_is_a_domain_package_that_depends_only_on_shared_values():
    allowed = ("kernel_workflow.rsi.trace", "kernel_workflow.rsi.discovery", "kernel_workflow.rsi.evidence")
    for module, dependencies in implementation_graph().items():
        if module.startswith("kernel_workflow.rsi.trace"):
            local = {value for value in dependencies if value.startswith("kernel_workflow.rsi")}
            assert all(value.startswith(allowed) for value in local), (module, local)
