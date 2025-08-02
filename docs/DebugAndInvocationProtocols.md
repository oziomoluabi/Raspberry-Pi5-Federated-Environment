# Debug & Invocation Protocols  
Standardized procedures for debugging and invocation...  
(Refer docs/DebugAndInvocationProtocols.md)

````markdown
# Debug & Invocation Protocols

## 1. Purpose  
Define standardized procedures for debugging each component of the All-in-One IoT Edge system and the precise invocation patterns for running its scripts, tasks, and tools. Ensures consistency, reproducibility, and rapid troubleshooting across local, remote, and container environments.

---

## 2. Scope  
Applies to:
- Federated Learning server (`server/`)  
- Edge client (`client/`)  
- TinyML, microTVM workflows  
- MATLAB Engine & Simulink integrations  
- VS Code tasks, CLI scripts, and containers

---

## 3. Debugging Protocols  

### 3.1 Local Debugging in VS Code  
1. **Open Workspace**  
   - `All_In_One_IoT_Edge.code-workspace`  
2. **Set Breakpoints**  
   - Python: click gutter in `.py` files (e.g. in `server/main.py.fit()`)  
   - C/C++: set in `.cpp` or generated TFLM code; ensure symbols are not stripped  
   - MATLAB: in `.m` files via the MATLAB extension  
3. **Launch Configurations** (`.vscode/launch.json`)  
   ```jsonc
   // Debug Federated Server
   {
     "name": "Debug Federated Server",
     "type": "python",
     "request": "launch",
     "program": "${workspaceFolder:server}/main.py",
     "console": "integratedTerminal"
   }
   // Debug Edge Client
   {
     "name": "Debug Edge Client",
     "type": "python",
     "request": "launch",
     "program": "${workspaceFolder:client}/main.py",
     "console": "integratedTerminal"
   }
````

4. **Attach Debugger**

   * For long-running processes, launch with `--wait-for-client` flag and use “Attach” config.

### 3.2 Remote Debugging via SSH

1. **Remote-SSH Setup**

   * In VS Code: Remote Explorer → configure `"remote.SSH.configFile"` → add `<user>@<pi-ip>`.
2. **Open Remote Workspace**

   * Connect to Pi and open the same `.code-workspace`.
3. **Launch Debug on Pi**

   * Use the same launch configs; VS Code will run the debuggee on the Pi.

### 3.3 Containerized Debugging

1. **DevContainer**

   * `.devcontainer/Dockerfile` builds environment with Python, TF-F, TFLite, TVM, MATLAB engine stub.
2. **Reopen in Container**

   * VS Code: Remote-Containers → Reopen in Container.
3. **Debug Inside Container**

   * Use same VS Code launch configs; they run inside the container.

### 3.4 C/C++ Debugging (TFLM & microTVM)

1. **Compile with Debug Symbols**

   * In `CMakeLists.txt`: `set(CMAKE_BUILD_TYPE Debug)`
2. **Launch GDB**

   * Use `gdb ${executable}` or VS Code C++ debug config:

     ```jsonc
     {
       "name": "Debug TFLM Module",
       "type": "cppdbg",
       "request": "launch",
       "program": "${workspaceFolder}/client/build/ae_module",
       "args": [],
       "cwd": "${workspaceFolder}/client"
     }
     ```
3. **microTVM RPC**

   * Start RPC server on target emulator: `tvm micro --run … --rpc …`
   * Attach `gdb` to the RPC-spawned process (via `target remote :9090`).

### 3.5 Logging & Diagnostics

* **Structured Logs** (`logging` module): set `logger = logging.getLogger()` with `JSONFormatter`.
* **Verbosity Levels**:

  * `DEBUG` for step-by-step tracing
  * `INFO` for normal operation
  * `WARN`/`ERROR` for issues
* **Log Files**: rotate via `TimedRotatingFileHandler`, stored under `logs/`.
* **MATLAB Logs**: capture Engine stdout by launching with `-logfile matlab.log`.

---

## 4. Invocation Protocols

### 4.1 Python Entry Points

| Component  | Command                                      | Notes                                      |
| ---------- | -------------------------------------------- | ------------------------------------------ |
| **Server** | `python3 server/main.py --config config.yml` | `--config` overrides default FL settings   |
| **Client** | `python3 client/main.py --mode live`         | `--mode=simulate` for local data emulation |

### 4.2 VS Code Tasks

Defined in `.vscode/tasks.json`:

```jsonc
{
  "label": "Run Federated Server",
  "type": "shell",
  "command": "python3",
  "args": ["main.py"],
  "options": { "cwd": "${workspaceFolder:server}" }
},
{
  "label": "Run Edge Client",
  "type": "shell",
  "command": "python3",
  "args": ["main.py"],
  "options": { "cwd": "${workspaceFolder:client}" }
}
```

### 4.3 MATLAB & Octave Invocation

* **MATLAB Engine**:

  ```python
  import matlab.engine
  eng = matlab.engine.start_matlab("-nojvm -nodesktop")
  eng.addpath('client/matlab', nargout=0)
  eng.env_preprocess(nargout=0)
  ```
* **Live Script**:

  ```bash
  matlab -batch "run('matlab/analysis.mlx')"
  ```
* **Octave Fallback**:

  ```python
  from oct2py import Oct2Py
  oc = Oct2Py()
  oc.run('client/matlab/env_preprocess.m')
  ```

### 4.4 microTVM Compilation & Invocation

1. **Compile**:

   ```python
   import tvm, tvm.micro
   # ... build code ...
   m = tvm.micro.compile_model(...)
   ```
2. **Run on Host**:

   ```bash
   python3 client/tvm_run.py --model ae-tvm-v1.0.tar.gz
   ```

### 4.5 Utility Scripts

* **Provisioning**: `scripts/provision_pis.sh <inventory.yml>`
* **Data Generation**: `scripts/gen_synthetic_data.py --output data/envdata.csv`

---

## 5. Best Practices

* **Consistent Launch**: always pass `--config` file to override defaults.
* **Isolate Environments**: use virtualenv or DevContainer to avoid dependency conflicts.
* **Reproducible Builds**: pin dependencies in `requirements.txt` and container images.
* **Clean State**: if debugging odd behavior, run `git clean -fdx` and rebuild.

---

## 6. Troubleshooting Tips

* **Stuck at “Waiting for client”**: ensure debug config’s `host`/`port` match launched process.
* **MATLAB Engine errors**: verify `which matlab` in `$PATH` and Python version compatibility.
* **microTVM build failures**: check `CROSS_COMPILE` and Zephyr SDK paths.
* **Container locale issues**: add `ENV LANG=C.UTF-8` in Dockerfile.

---

*Prepared by \[Your Name], Software Architect – August 2, 2025*

```
::contentReference[oaicite:0]{index=0}
```
