# Experiment Logs

Performance optimization experiment tracking. Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch).

## Structure

```
experiments/
├── README.md          # This file
├── baseline.txt       # Reference: C libjpeg-turbo benchmark results
├── <target>/          # One subfolder per optimization area
│   ├── log.tsv        # Experiment ledger (append-only)
│   ├── notes.md       # (optional) Hypotheses, reference links, dead-end rationale
│   └── profile.txt    # (optional) Profiler output snapshots
└── ...
```

Current targets: `pipeline/`, `huffman/`, `idct/`, `upsample/`.

## Per-Target TSV Format

Each optimization area gets its own subfolder with a `log.tsv`. When switching targets, **read only that subfolder** — not the others. This prevents context pollution and keeps focus.

**Columns** (tab-separated):

```
commit	time_us	status	description
```

| Column | Type | Description |
|--------|------|-------------|
| `commit` | string | Short git commit hash (7 chars), or `uncommitted` |
| `time_us` | float | Benchmark result: decode_640x480 in microseconds |
| `status` | enum | `keep` = improved, `discard` = reverted, `crash` = failed to compile/run |
| `description` | string | What was tried and why it worked/failed |

## Status Values

- **keep**: Benchmark improved (or equal with simpler code). Commit stays.
- **discard**: Benchmark regressed. Changes reverted via `git checkout`.
- **crash**: Compilation error or runtime panic. Log what went wrong.

## Rules

1. **Record every experiment** — including failures. Failures are data.
2. **One TSV per target** — when optimizing IDCT, only read `idct.tsv`. When optimizing Huffman, only read `huffman.tsv`. Prevents context bloat.
3. **Description must explain WHY** — not just "tried X" but "tried X because Y; failed because Z" or "tried X because Y; worked because Z".
4. **Benchmark before AND after** — always record the before baseline in the description if it's not the previous row.
5. **Never delete rows** — append only. The history of failures is as valuable as successes.

## Workflow

```
1. Read the relevant <target>.tsv (only that one)
2. Profile to find the hotspot
3. Form a hypothesis (what to change and why)
4. Make the change
5. cargo test (must pass)
6. cargo bench -- decode_640x480 (record result)
7. If better: commit + append row with status=keep
   If worse:  revert + append row with status=discard
   If crash:  append row with status=crash + error summary
8. Repeat
```
