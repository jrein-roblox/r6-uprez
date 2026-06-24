# RoMotion — Studio plugin setup

RoMotion is a generative-animation plugin for Roblox Studio: write a text
prompt (and optionally place pose constraints in the viewport), and it
synthesizes a full-body R15 animation using [Kimodo](https://github.com/nv-tlabs/kimodo)
and plays it on your rig.

It has two parts:

1. **Backend server** (`server/`) — a FastAPI app that wraps Kimodo. It loads
   the model once, keeps it warm in memory, and serves generation requests on
   `http://localhost:8787`.
2. **Studio plugin** (`plugin/`) — a Rojo project (vanilla Luau). It runs inside
   Studio and talks to the server over HTTP.

```
Roblox Studio ──HTTP──► RoMotion server (FastAPI) ──in-process──► Kimodo model
  (plugin)              localhost:8787
```

Both run on the same machine. The plugin always points at `localhost:8787`.

---

## Prerequisites

| Tool | Why | Notes |
| --- | --- | --- |
| **Kimodo** + its Python venv | The server imports `kimodo` + `torch` and runs the model in-process | See [Kimodo](https://github.com/nv-tlabs/kimodo). The venv must have `torch`, `kimodo`, and the model weights available (Kimodo caches them on first run). |
| **Python ≥ 3.10** | Server runtime | Use the **same interpreter that has Kimodo** (see below) — not system python. |
| **[Rojo](https://rojo.space) ≥ 7.x** | Builds the plugin `.rbxm` | `cargo install rojo` or `aftman add rojo-rbx/rojo`. Verified with Rojo 7.6.1. |
| **Roblox Studio** | Runs the plugin | macOS or Windows. |

> **Hardware:** the server runs on CUDA (NVIDIA), Apple Silicon MPS, or CPU —
> it auto-detects (`kimodo_warm._pick_device`). CPU works but is slow.

---

## 1. Install Kimodo

Follow the [Kimodo](https://github.com/nv-tlabs/kimodo) setup to create a venv
and install it (e.g. with `uv` or `pip -e .`). You should end up with a Python
interpreter where this works:

```bash
/path/to/kimodo/.venv/bin/python -c "import kimodo, torch; print('ok', torch.__version__)"
```

The server reuses **this same venv** because it needs `torch` + `kimodo` in
process. The server's own deps (`fastapi`, `uvicorn`, `numpy`, `scipy` — see
`server/pyproject.toml`) must also be importable from it:

```bash
/path/to/kimodo/.venv/bin/python -m pip install "fastapi>=0.104" "uvicorn[standard]>=0.24" "numpy>=1.24" "scipy>=1.10"
```

> The Kimodo install is currently external (it lives in its own repo/venv, not
> vendored here). Note its venv python path — you'll use it in the next step.

## 2. Run the backend server

```bash
cd r6-uprez/server
/path/to/kimodo/.venv/bin/python main.py
```

On startup it prints:

```
[kimodo_warm] Loading model <name> on <device> (one-time)...
[kimodo_warm] Model ready: <name>
```

The first load is slow (model + text encoder); subsequent generations are fast.
The server listens on `http://0.0.0.0:8787`. Verify it's up:

```bash
curl http://localhost:8787/health   # -> {"status":"ok"}
```

Leave this running while you use the plugin.

## 3. Build & install the plugin

Rojo builds the plugin straight into your local Studio plugins folder.

**macOS:**

```bash
cd r6-uprez/plugin
rojo build -o "$HOME/Documents/Roblox/Plugins/RoMotion.rbxm"
```

**Windows:**

```powershell
cd r6-uprez\plugin
rojo build -o "$env:LOCALAPPDATA\Roblox\Plugins\RoMotion.rbxm"
```

Studio picks up new/changed plugin files automatically — just rebuild and the
plugin reloads. (Restart Studio if it doesn't.)

## 4. Enable HTTP requests in Studio

The plugin calls `localhost:8787` via `HttpService`, which is off by default.
Enable it in the place you're working in, either:

- **Game Settings → Security → Allow HTTP Requests** (toggle on), or
- run this once in the **Command Bar**:

  ```lua
  game:GetService("HttpService").HttpEnabled = true
  ```

---

## Using it

1. Make sure the server is running (step 2) and HTTP is enabled (step 4).
2. Open the RoMotion widget from the Plugins toolbar.
3. Select an R15 rig in the workspace (it auto-detects the rig).
4. Type a prompt on the timeline (e.g. *"a person waves hello"*), set a
   duration, and hit **Generate**. The animation builds and plays on the rig.
5. *(Optional)* Add constraints with the per-effector **+** buttons — place the
   gizmo in the viewport to pin a hand/foot/hips/look pose at a frame.
   Double-click a timeline diamond to toggle **hard** (exact pin) vs **soft**
   (model influence only).

---

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `No module named 'torch'` / `kimodo` on server start | You're running with the wrong interpreter. Use Kimodo's venv python, not system `python3`. |
| Plugin error: `Http requests are not enabled` | Do step 4 (enable HTTP requests in the place). |
| Plugin can't reach the server / connection refused | The server isn't running, or it's not on `localhost:8787`. Check the server console and `curl http://localhost:8787/health`. |
| Generation is very slow | Running on CPU. Use a CUDA or Apple-Silicon (MPS) machine; check the device in the `[kimodo_warm] Loading model ... on <device>` line. |
| Plugin doesn't appear in Studio | Confirm the `.rbxm` landed in the plugins folder (step 3) and restart Studio. |

## Notes & limitations

- **No persistence yet** — closing the plugin loses prompts/constraints.
- Server URL is hardcoded to `localhost:8787` in
  `plugin/src/Utils/Constants.lua` (`Constants.SERVER_URL`). Change it there if
  you run the server elsewhere, then rebuild.
- For the headless command-line pipeline (re-imagine an existing asset, or batch
  prompt synthesis), see [`README.md`](README.md) and [`PIPELINE.md`](PIPELINE.md).
</content>
</invoke>
