# r6-uprez

Re-imagine Roblox R6/R15 animations as full-body R15 motion via Kimodo, or
synthesize R15 animations from text prompts directly.

> **Want the Studio plugin?** RoMotion lets you generate animations live inside
> Roblox Studio from a text prompt + viewport constraints. See
> [`ROMOTION.md`](ROMOTION.md) for plugin + server setup.

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

## Asset-id pipeline

Re-imagine an existing R6/R15 animation. Rig is auto-detected from the clip's
bone names.

```
python3 python/pipeline.py --asset-id <id> --out work --name <name> --prompt "<text>"
```

Test assets:
- 180426354 — R6 walk (looped)
- 182435998 — R6 overhead wave (looped, exercises torso bob + arm singularity)
- 180436334 — R6 short dance (looped)
- 129423131 — R6 motion (non-loop)

## Prompt pipeline

Skip pose extraction entirely and synthesize from text. Output also lands at
`work/<name>/r15.rbxm`.

```
uv run --with numpy --with scipy python/prompt_pipeline.py \
    --prompt "a person waves hello" --out work --name wave --duration 3.0
```

Looped clip (two-pass synthesis pins start and end pose):

```
python/prompt_pipeline.py --prompt "a person dances" --name dance \
    --out work --duration 4.0 --loop
```

Chained prompts (kimodo splits on `.`, one duration per segment):

```
python/prompt_pipeline.py \
    --prompt "a person waves hello. then they sit down." \
    --duration "2.0 3.0" --out work --name wave-then-sit
```

See `PIPELINE.md` for stage-by-stage details, all flags, defaults, and known
issues for both pipelines.
