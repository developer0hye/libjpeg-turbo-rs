Read `CLAUDE.md`, `METHODOLOGY.md` and `CONTRIBUTING.md` before you start, and
follow them literally. The points below are the ones an unattended run gets
wrong most often; they do not replace those documents.

- Read `docs/LAST_MILE.md` first — every session, even for a small task. Then
  open the one phase file under `docs/last_mile/` that covers the gap you are
  touching, and no others.
- Work in a git worktree on a `<type>/<short-description>` branch cut from
  current `main`. Never `git checkout` in the main working tree — other
  sessions use it.
- Run `git config core.hooksPath .githooks` inside the worktree so the
  pre-commit hook runs `cargo fmt --check` and `cargo clippy --lib -- -D warnings`
  before every commit. Local clippy (aarch64) and CI clippy (x86_64) differ
  through `#[cfg(target_arch)]` blocks — fix warnings for both.
- Test-driven: the failing test comes first. Where a C contract exists,
  cross-validate against `djpeg`/`cjpeg`/`jpegtran` rather than against our own
  previous output — byte-exact where the format allows it.
- Sign off every commit (`git commit -s`).
- **Scope honestly.** These issues are programs, not one-line defects. If the
  whole issue cannot be finished and verified in one session, land the first
  coherent, independently valuable milestone, comment on the issue stating
  exactly what landed and what remains, and leave the issue open. A landed
  milestone is a successful session.
- Keep the tracking docs true **before** the pull request merges: the phase
  file's section state (`OPEN` / `PARTIAL: …` / `CLOSED YYYY-MM-DD` with the
  command that proves it), the OPEN Items table in `docs/LAST_MILE.md`, and the
  `docs/FEATURE_PARITY.md` checkbox and `docs/C_API_REFERENCE.md` status when a
  canonical mapping changed. Never mark a gap `CLOSED` that is only partly done
  — that is what the `PARTIAL` state is for.
- Anything you discover mid-work that is not already tracked gets filed as its
  own gap, and its own issue, before the pull request that surfaced it merges.
- Never weaken, skip, or `#[ignore]` a test to make CI green.
