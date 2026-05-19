# r6-uprez

Re-imagine Roblox R6/R15 animations as full-body R15 motion via Kimodo.

```
asset id ──► extract_pose.lua  ──► pose.json
            (roblox-cli)
                │
                ▼
        roblox_to_kimodo.py ──► constraints.json
                │
                ▼
            run_kimodo.py ──► generated.bvh
                │
                ▼
        export_r15 + build_rbxm ──► r15.rbxm
```

Run end-to-end (rig is auto-detected from the clip's bone names):
```
python3 python/pipeline.py --asset-id <id> --out work --name <name> --prompt "<text>"
```

Test assets that exercise the pipeline:
- 180426354 — R6 walk (looped)
- 182435998 — R6 overhead wave (looped, exercises torso bob + arm singularity)
- 180436334 — R6 short dance (looped)
- 129423131 — R6 motion (non-loop)

See `PIPELINE.md` for stage-by-stage details, all flags, defaults, and known issues.
