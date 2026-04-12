//! Contained Python runtime abstraction for ARP (issue #16).
//!
//! Manages portable Python virtual environments per experiment with:
//! - Reproducible venv creation from a [`PythonRuntimeSpec`]
//! - Traceable task execution with event-ledger integration
//! - Lifecycle states: `Active` → `Deprecated` → `Deleted`
//!
//! File layout on disk:
//! ```text
//! runtimes/python/
//!   python_runtimes.json          ← RuntimeRegistry
//!   {id}/
//!     create.log                  ← full creation transcript
//!     requirements.lock           ← pip freeze output
//!     bin/python  (or Scripts\python.exe on Windows)
//!     …(venv contents)
//!
//! runs/{experiment_id}/{run_id}/
//!   python_tasks/{task_id}/
//!     stdout.log
//!     stderr.log
//!     meta.json
//! ```

use anyhow::{Context, Result, anyhow, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};

use crate::event_ledger::EventLedger;
use crate::reproducibility::now_rfc3339_utc;

// ─── Counters for unique IDs ───────────────────────────────────────────────

static RUNTIME_COUNTER: AtomicU64 = AtomicU64::new(0);
static TASK_COUNTER: AtomicU64 = AtomicU64::new(0);

fn sha256_prefix8(data: &str) -> String {
    let mut h = Sha256::new();
    h.update(data.as_bytes());
    h.finalize().iter().take(4).map(|b| format!("{:02x}", b)).collect()
}

fn new_runtime_id(label: &str) -> String {
    let nonce = RUNTIME_COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("pyrt_{}", sha256_prefix8(&format!("{label}{ts}{nonce}")))
}

fn new_task_id(runtime_id: &str, entrypoint: &str) -> String {
    let nonce = TASK_COUNTER.fetch_add(1, Ordering::Relaxed);
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos();
    format!("task_{}", sha256_prefix8(&format!("{runtime_id}{entrypoint}{ts}{nonce}")))
}

// ─── Core types ────────────────────────────────────────────────────────────

/// Lifecycle state of a [`PythonRuntime`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum PythonRuntimeState {
    #[default]
    Active,
    /// No new experiments can be attached; existing runs still reference it.
    Deprecated,
    /// Directory removed; manifest entry retained for historical traceability.
    Deleted,
}

/// Declarative specification for creating a portable Python venv.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonRuntimeSpec {
    /// Path or selector for the system Python interpreter (e.g. `"python3.11"`).
    pub base_interpreter: String,
    /// PyPI-style requirement strings, e.g. `["numpy>=1.26", "psychopy==2024.1.0"]`.
    pub requirements: Vec<String>,
    /// Optional commands run inside the runtime after `pip install` (e.g. verification).
    /// Each string is split on whitespace — do not use shell metacharacters.
    #[serde(default)]
    pub post_install_commands: Vec<String>,
}

/// A managed Python virtual environment with full provenance metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonRuntime {
    /// Stable internal id, e.g. `pyrt_1a2b3c4d`.
    pub id: String,
    /// Human-readable label, e.g. `"StroopTask-Python-3.11"`.
    pub label: String,
    /// Actual Python version string resolved at creation time (e.g. `"3.11.9"`).
    pub python_version: String,
    /// Base directory of this venv. `None` when `state` is `Deleted`.
    pub root_path: Option<PathBuf>,
    pub created_at: String,
    /// Identifier of the user/process that created the runtime.
    pub created_by: String,
    pub spec: PythonRuntimeSpec,
    #[serde(default)]
    pub state: PythonRuntimeState,
}

/// Lightweight reference embedded in a run manifest for traceability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonRuntimeRef {
    pub runtime_id: String,
    pub resolved_python_version: String,
    /// SHA-256 hex digest of `requirements.lock`.
    pub requirements_lock_hash: String,
}

/// Configuration for a single Python task invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonTaskConfig {
    /// ID of the [`PythonRuntime`] to use.
    pub runtime_id: String,
    /// Path to the Python script (relative to `working_dir`).
    pub entrypoint: PathBuf,
    /// CLI arguments passed to the script.
    #[serde(default)]
    pub args: Vec<String>,
    /// Additional environment variable overrides merged with ARP defaults.
    #[serde(default)]
    pub env: HashMap<String, String>,
    /// Working directory; defaults to `{run_dir}/python_tasks/{task_id}`.
    pub working_dir: Option<PathBuf>,
}

/// Sidecar metadata written to `{task_dir}/meta.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PythonTaskMeta {
    pub task_id: String,
    pub runtime_id: String,
    /// Full command as a list of strings (python binary + entrypoint + args).
    pub command: Vec<String>,
    /// Env vars injected into the child process (user overrides + ARP vars).
    pub env_overrides: HashMap<String, String>,
    pub exit_code: i32,
    pub started_at: String,
    pub finished_at: String,
    pub stdout_log: PathBuf,
    pub stderr_log: PathBuf,
}

// ─── Registry ──────────────────────────────────────────────────────────────

/// Persistent registry of all managed Python runtimes for this workspace.
///
/// Stored as `runtimes/python/python_runtimes.json`.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct RuntimeRegistry {
    pub runtimes: Vec<PythonRuntime>,
}

impl RuntimeRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// Load from a JSON file, returning an empty registry if the file does not exist.
    pub fn load(path: &Path) -> Result<Self> {
        if !path.exists() {
            return Ok(Self::new());
        }
        let raw = fs::read_to_string(path)
            .with_context(|| format!("read runtime registry {}", path.display()))?;
        serde_json::from_str(&raw).context("parse python_runtimes.json")
    }

    /// Persist to a JSON file (pretty-printed).
    pub fn save(&self, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)
                .with_context(|| format!("create registry parent dir {}", parent.display()))?;
        }
        let json = serde_json::to_string_pretty(self).context("serialize runtime registry")?;
        fs::write(path, json)
            .with_context(|| format!("write runtime registry {}", path.display()))
    }

    pub fn find(&self, id: &str) -> Option<&PythonRuntime> {
        self.runtimes.iter().find(|r| r.id == id)
    }

    pub fn find_mut(&mut self, id: &str) -> Option<&mut PythonRuntime> {
        self.runtimes.iter_mut().find(|r| r.id == id)
    }
}

// ─── Platform helpers ──────────────────────────────────────────────────────

fn venv_python(root: &Path) -> PathBuf {
    if cfg!(windows) {
        root.join("Scripts").join("python.exe")
    } else {
        root.join("bin").join("python")
    }
}

fn venv_bin_dir(root: &Path) -> PathBuf {
    if cfg!(windows) {
        root.join("Scripts")
    } else {
        root.join("bin")
    }
}

// ─── Runtime creation ──────────────────────────────────────────────────────

/// Create a new managed Python venv from `spec`, recording all output to `create.log`.
///
/// Returns the fully-populated [`PythonRuntime`] ready to be pushed into a [`RuntimeRegistry`].
///
/// # Errors
/// Returns an error if the base interpreter is missing, venv creation fails, or pip fails.
pub fn create_runtime(
    spec: PythonRuntimeSpec,
    label: &str,
    created_by: &str,
    runtimes_dir: &Path,
) -> Result<PythonRuntime> {
    let id = new_runtime_id(label);
    let root_path = runtimes_dir.join(&id);
    fs::create_dir_all(&root_path)
        .with_context(|| format!("create runtime dir {}", root_path.display()))?;

    let log_path = root_path.join("create.log");
    let mut log = fs::File::create(&log_path)
        .with_context(|| format!("create create.log at {}", log_path.display()))?;

    let created_at = now_rfc3339_utc();
    writeln!(log, "[{created_at}] Creating runtime {id} (label: {label})")?;
    writeln!(log, "base_interpreter: {}", spec.base_interpreter)?;
    writeln!(log, "requirements: {:?}", spec.requirements)?;

    // 1. Create venv
    writeln!(log, "\n--- venv creation ---")?;
    let venv_out = Command::new(&spec.base_interpreter)
        .args(["-m", "venv", root_path.to_str().unwrap_or(&id)])
        .output()
        .with_context(|| format!("spawn `{} -m venv`", spec.base_interpreter))?;
    log.write_all(&venv_out.stdout)?;
    log.write_all(&venv_out.stderr)?;
    if !venv_out.status.success() {
        bail!(
            "venv creation failed (exit {}): {}",
            venv_out.status,
            String::from_utf8_lossy(&venv_out.stderr)
        );
    }

    // 2. Resolve actual Python version from the freshly created venv
    let python_binary = venv_python(&root_path);
    let ver_out = Command::new(&python_binary)
        .arg("--version")
        .output()
        .with_context(|| format!("query python version at {}", python_binary.display()))?;
    // CPython ≥3.4 prints to stdout; older versions print to stderr
    let ver_stdout = String::from_utf8_lossy(&ver_out.stdout);
    let ver_stderr = String::from_utf8_lossy(&ver_out.stderr);
    let python_version = ver_stdout
        .trim()
        .strip_prefix("Python ")
        .or_else(|| ver_stderr.trim().strip_prefix("Python "))
        .unwrap_or("unknown")
        .to_string();
    writeln!(log, "resolved Python version: {python_version}")?;

    // 3. Upgrade pip silently (best-effort)
    let pip_binary = venv_bin_dir(&root_path).join(if cfg!(windows) { "pip.exe" } else { "pip" });
    writeln!(log, "\n--- pip upgrade ---")?;
    let pip_upgrade = Command::new(&pip_binary)
        .args(["install", "--upgrade", "pip"])
        .output()
        .context("upgrade pip")?;
    log.write_all(&pip_upgrade.stdout)?;
    log.write_all(&pip_upgrade.stderr)?;

    // 4. Install requirements (single pip invocation for speed)
    if !spec.requirements.is_empty() {
        writeln!(log, "\n--- pip install requirements ---")?;
        let mut pip_args = vec!["install"];
        let req_refs: Vec<&str> = spec.requirements.iter().map(String::as_str).collect();
        pip_args.extend_from_slice(&req_refs);
        let pip_out = Command::new(&pip_binary)
            .args(&pip_args)
            .output()
            .context("pip install requirements")?;
        log.write_all(&pip_out.stdout)?;
        log.write_all(&pip_out.stderr)?;
        if !pip_out.status.success() {
            bail!(
                "pip install failed (exit {}): {}",
                pip_out.status,
                String::from_utf8_lossy(&pip_out.stderr)
            );
        }
    }

    // 5. Run post-install verification commands
    for cmd_str in &spec.post_install_commands {
        writeln!(log, "\n--- post-install: {cmd_str} ---")?;
        // Split on whitespace only; issue requires "shell-safe" commands with no metacharacters
        let parts: Vec<&str> = cmd_str.split_whitespace().collect();
        let (prog, args) = parts.split_first().unwrap_or((&"python", &[]));
        let cmd_out = Command::new(prog)
            .args(args)
            .env("PATH", format!("{}:{}", venv_bin_dir(&root_path).display(), std::env::var("PATH").unwrap_or_default()))
            .output()
            .with_context(|| format!("post-install command: {cmd_str}"))?;
        log.write_all(&cmd_out.stdout)?;
        log.write_all(&cmd_out.stderr)?;
    }

    // 6. Capture requirements.lock via pip freeze
    writeln!(log, "\n--- pip freeze (requirements.lock) ---")?;
    let freeze_out = Command::new(&pip_binary)
        .arg("freeze")
        .output()
        .context("pip freeze")?;
    let lock_content = String::from_utf8_lossy(&freeze_out.stdout).to_string();
    let lock_path = root_path.join("requirements.lock");
    fs::write(&lock_path, &lock_content)
        .with_context(|| format!("write requirements.lock at {}", lock_path.display()))?;
    log.write_all(lock_content.as_bytes())?;

    let done_at = now_rfc3339_utc();
    writeln!(log, "\n[{done_at}] Runtime {id} created successfully.")?;

    Ok(PythonRuntime {
        id,
        label: label.to_string(),
        python_version,
        root_path: Some(root_path),
        created_at,
        created_by: created_by.to_string(),
        spec,
        state: PythonRuntimeState::Active,
    })
}

// ─── Runtime reference ─────────────────────────────────────────────────────

/// Build a [`PythonRuntimeRef`] from a live runtime (reads `requirements.lock` from disk).
pub fn runtime_ref(runtime: &PythonRuntime) -> Result<PythonRuntimeRef> {
    let root = runtime
        .root_path
        .as_deref()
        .ok_or_else(|| anyhow!("runtime {} has been deleted (no root_path)", runtime.id))?;
    let lock_path = root.join("requirements.lock");
    let lock_content = fs::read_to_string(&lock_path)
        .with_context(|| format!("read requirements.lock at {}", lock_path.display()))?;
    let mut h = Sha256::new();
    h.update(lock_content.as_bytes());
    let hash: String = h.finalize().iter().map(|b| format!("{:02x}", b)).collect();
    Ok(PythonRuntimeRef {
        runtime_id: runtime.id.clone(),
        resolved_python_version: runtime.python_version.clone(),
        requirements_lock_hash: hash,
    })
}

// ─── Task execution ────────────────────────────────────────────────────────

fn resolve_python_binary(runtime: &PythonRuntime) -> Result<PathBuf> {
    if runtime.state == PythonRuntimeState::Deleted {
        bail!("runtime {} is Deleted — cannot execute tasks", runtime.id);
    }
    let root = runtime
        .root_path
        .as_deref()
        .ok_or_else(|| anyhow!("runtime {} has no root_path", runtime.id))?;
    let python = venv_python(root);
    if !python.exists() {
        bail!(
            "runtime {} Python binary not found at {}",
            runtime.id,
            python.display()
        );
    }
    Ok(python)
}

/// Spawn a Python script inside a managed runtime with full traceability.
///
/// - Captures stdout/stderr to `{task_dir}/stdout.log` and `stderr.log`.
/// - Writes `{task_dir}/meta.json`.
/// - Emits `python_task.started` and `python_task.finished` events into `ledger`.
///
/// `run_dir` is the active `runs/{experiment_id}/{run_id}` directory.
pub fn launch_task(
    config: &PythonTaskConfig,
    runtime: &PythonRuntime,
    run_dir: &Path,
    experiment_id: &str,
    run_id: &str,
    ledger: &EventLedger,
) -> Result<PythonTaskMeta> {
    let python = resolve_python_binary(runtime)?;
    let task_id = new_task_id(&config.runtime_id, &config.entrypoint.to_string_lossy());

    // Task output directory
    let task_dir = run_dir.join("python_tasks").join(&task_id);
    fs::create_dir_all(&task_dir)
        .with_context(|| format!("create task dir {}", task_dir.display()))?;

    let working_dir = config.working_dir.clone().unwrap_or_else(|| task_dir.clone());
    let stdout_log = task_dir.join("stdout.log");
    let stderr_log = task_dir.join("stderr.log");
    let events_path = run_dir.join("events.jsonl");

    // Build venv PATH prefix (no global venv activation; env-vars only)
    let venv_root = runtime.root_path.as_deref().expect("validated by resolve_python_binary");
    let venv_bin = venv_bin_dir(venv_root);
    let path_with_venv = format!(
        "{}:{}",
        venv_bin.display(),
        std::env::var("PATH").unwrap_or_default()
    );

    // Recorded env for meta.json: user overrides + ARP injected vars
    let mut recorded_env = config.env.clone();
    recorded_env.insert("ARP_EXPERIMENT_ID".to_string(), experiment_id.to_string());
    recorded_env.insert("ARP_RUN_ID".to_string(), run_id.to_string());
    recorded_env.insert(
        "ARP_EVENTS_PATH".to_string(),
        events_path.to_string_lossy().to_string(),
    );

    // Full command for meta.json
    let mut cmd_parts = vec![python.to_string_lossy().to_string()];
    cmd_parts.push(config.entrypoint.to_string_lossy().to_string());
    cmd_parts.extend(config.args.iter().cloned());

    // Emit started event before spawning
    let started_at = now_rfc3339_utc();
    ledger.append_with_hashes(
        "python_task.started",
        None,
        None,
        &config.runtime_id,
        &task_id,
        serde_json::json!({
            "task_id": &task_id,
            "runtime_id": &config.runtime_id,
            "entrypoint": config.entrypoint,
            "args": config.args,
            "working_dir": working_dir,
        }),
    )?;

    // Spawn child process
    let stdout_file = fs::File::create(&stdout_log)
        .with_context(|| format!("create stdout.log at {}", stdout_log.display()))?;
    let stderr_file = fs::File::create(&stderr_log)
        .with_context(|| format!("create stderr.log at {}", stderr_log.display()))?;

    let mut child = Command::new(&python)
        .arg(&config.entrypoint)
        .args(&config.args)
        .current_dir(&working_dir)
        // User-defined overrides (applied first, lower priority)
        .envs(&config.env)
        // Venv environment (overrides system defaults)
        .env("PATH", &path_with_venv)
        .env("VIRTUAL_ENV", venv_root)
        // ARP-specific vars (non-overridable by user config)
        .env("ARP_EXPERIMENT_ID", experiment_id)
        .env("ARP_RUN_ID", run_id)
        .env("ARP_EVENTS_PATH", events_path.as_os_str())
        .stdout(Stdio::from(stdout_file))
        .stderr(Stdio::from(stderr_file))
        .spawn()
        .with_context(|| format!("spawn python task {task_id}"))?;

    let status = child
        .wait()
        .with_context(|| format!("wait for python task {task_id}"))?;
    let exit_code = status.code().unwrap_or(-1);
    let finished_at = now_rfc3339_utc();

    // Write meta.json sidecar
    let meta = PythonTaskMeta {
        task_id: task_id.clone(),
        runtime_id: config.runtime_id.clone(),
        command: cmd_parts,
        env_overrides: recorded_env,
        exit_code,
        started_at: started_at.clone(),
        finished_at: finished_at.clone(),
        stdout_log: stdout_log.clone(),
        stderr_log: stderr_log.clone(),
    };
    let meta_path = task_dir.join("meta.json");
    fs::write(
        &meta_path,
        serde_json::to_string_pretty(&meta).context("serialize task meta")?,
    )
    .with_context(|| format!("write meta.json at {}", meta_path.display()))?;

    // Emit finished event
    ledger.append_with_hashes(
        "python_task.finished",
        None,
        None,
        &task_id,
        &exit_code.to_string(),
        serde_json::json!({
            "task_id": &task_id,
            "runtime_id": &config.runtime_id,
            "exit_code": exit_code,
            "started_at": &started_at,
            "finished_at": &finished_at,
            "stdout_log": &stdout_log,
            "stderr_log": &stderr_log,
            "meta_json": &meta_path,
        }),
    )?;

    Ok(meta)
}

// ─── Lifecycle management ──────────────────────────────────────────────────

/// Mark a runtime as `Deprecated` — no new experiments can be attached to it.
pub fn deprecate_runtime(registry: &mut RuntimeRegistry, id: &str) -> Result<()> {
    let rt = registry
        .find_mut(id)
        .ok_or_else(|| anyhow!("runtime {id} not found in registry"))?;
    if rt.state == PythonRuntimeState::Deleted {
        bail!("runtime {id} is already Deleted");
    }
    rt.state = PythonRuntimeState::Deprecated;
    Ok(())
}

/// Physically remove the runtime directory and mark the registry entry as `Deleted`.
///
/// The manifest entry is retained for historical traceability; `root_path` is set to `None`.
/// An error is returned if the runtime is already deleted.
pub fn delete_runtime(registry: &mut RuntimeRegistry, id: &str) -> Result<()> {
    let rt = registry
        .find_mut(id)
        .ok_or_else(|| anyhow!("runtime {id} not found in registry"))?;
    if rt.state == PythonRuntimeState::Deleted {
        bail!("runtime {id} is already Deleted");
    }
    if let Some(root) = rt.root_path.take() {
        if root.exists() {
            fs::remove_dir_all(&root)
                .with_context(|| format!("remove runtime dir {}", root.display()))?;
        }
    }
    rt.state = PythonRuntimeState::Deleted;
    Ok(())
}

// ─── Canonical paths ───────────────────────────────────────────────────────

/// Default base directory for managed Python runtimes: `runtimes/python`.
pub fn default_runtimes_dir() -> PathBuf {
    PathBuf::from("runtimes").join("python")
}

/// Default path to the runtime registry file: `runtimes/python/python_runtimes.json`.
pub fn default_registry_path() -> PathBuf {
    default_runtimes_dir().join("python_runtimes.json")
}

// ─── Tests ─────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn runtime_id_is_prefixed_and_unique() {
        let a = new_runtime_id("test-label");
        let b = new_runtime_id("test-label");
        assert!(a.starts_with("pyrt_"), "id should start with 'pyrt_': {a}");
        assert_ne!(a, b, "consecutive IDs should differ");
    }

    #[test]
    fn task_id_is_prefixed_and_unique() {
        let a = new_task_id("pyrt_aabbccdd", "tasks/run.py");
        let b = new_task_id("pyrt_aabbccdd", "tasks/run.py");
        assert!(a.starts_with("task_"), "id should start with 'task_': {a}");
        assert_ne!(a, b, "consecutive IDs should differ");
    }

    #[test]
    fn registry_roundtrip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("python_runtimes.json");

        let mut reg = RuntimeRegistry::new();
        reg.runtimes.push(PythonRuntime {
            id: "pyrt_test1234".to_string(),
            label: "Test Runtime".to_string(),
            python_version: "3.11.9".to_string(),
            root_path: Some(PathBuf::from("/tmp/test")),
            created_at: "2026-04-12T00:00:00Z".to_string(),
            created_by: "test".to_string(),
            spec: PythonRuntimeSpec {
                base_interpreter: "python3.11".to_string(),
                requirements: vec!["numpy>=1.26".to_string()],
                post_install_commands: vec![],
            },
            state: PythonRuntimeState::Active,
        });
        reg.save(&path).expect("save");

        let loaded = RuntimeRegistry::load(&path).expect("load");
        assert_eq!(loaded.runtimes.len(), 1);
        assert_eq!(loaded.runtimes[0].id, "pyrt_test1234");
        assert_eq!(loaded.runtimes[0].python_version, "3.11.9");
    }

    #[test]
    fn registry_load_missing_file_returns_empty() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("does_not_exist.json");
        let reg = RuntimeRegistry::load(&path).expect("load");
        assert!(reg.runtimes.is_empty());
    }

    #[test]
    fn deprecate_then_delete_lifecycle() {
        let mut reg = RuntimeRegistry::new();
        reg.runtimes.push(PythonRuntime {
            id: "pyrt_lifecycle".to_string(),
            label: "Lifecycle Test".to_string(),
            python_version: "3.11.0".to_string(),
            root_path: None, // no actual dir for this test
            created_at: now_rfc3339_utc(),
            created_by: "test".to_string(),
            spec: PythonRuntimeSpec {
                base_interpreter: "python3".to_string(),
                requirements: vec![],
                post_install_commands: vec![],
            },
            state: PythonRuntimeState::Active,
        });

        deprecate_runtime(&mut reg, "pyrt_lifecycle").expect("deprecate");
        assert_eq!(reg.runtimes[0].state, PythonRuntimeState::Deprecated);

        delete_runtime(&mut reg, "pyrt_lifecycle").expect("delete");
        assert_eq!(reg.runtimes[0].state, PythonRuntimeState::Deleted);
        assert!(reg.runtimes[0].root_path.is_none());
    }

    #[test]
    fn double_delete_returns_error() {
        let mut reg = RuntimeRegistry::new();
        reg.runtimes.push(PythonRuntime {
            id: "pyrt_deldel".to_string(),
            label: "Double Delete".to_string(),
            python_version: "3.11.0".to_string(),
            root_path: None,
            created_at: now_rfc3339_utc(),
            created_by: "test".to_string(),
            spec: PythonRuntimeSpec {
                base_interpreter: "python3".to_string(),
                requirements: vec![],
                post_install_commands: vec![],
            },
            state: PythonRuntimeState::Deleted,
        });
        assert!(delete_runtime(&mut reg, "pyrt_deldel").is_err());
    }
}
