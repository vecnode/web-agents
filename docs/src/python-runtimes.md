# Python Runtime Management

Python environments are first-class, reproducible artifacts.

## What is managed

Each runtime is represented by `PythonRuntime` with metadata, lifecycle state, and declarative creation spec.

```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonRuntime {
    pub id: String,
    pub label: String,
    pub python_version: String,
    pub root_path: Option<PathBuf>,
    pub created_at: String,
    pub created_by: String,
    pub spec: PythonRuntimeSpec,
    pub state: PythonRuntimeState,
}
```

States:

- Active
- Deprecated
- Deleted

## Creation flow

`create_runtime` performs:

1. `python -m venv`
2. pip upgrade
3. dependency install
4. optional post-install commands
5. `pip freeze` snapshot into `requirements.lock`

All outputs are written to `create.log` in the runtime directory.

## Task execution flow

`launch_task` runs Python scripts with full run linkage:

- `stdout.log`
- `stderr.log`
- `meta.json`
- ledger events (`python_task.started`, `python_task.finished`)

This design allows deterministic reconstruction of the runtime used for each run.
