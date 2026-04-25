# Next Session Plan — libjpeg-turbo-rs

작성일: 2026-04-25  |  상태: main HEAD `dad30be`, 모든 워크플로우 green

## 0. 매 세션 Pre-flight (필수)

세션 시작 직후 항상 실행:

```bash
git status --short && git log --oneline -5
gh run list --limit 3 --branch main --json workflowName,conclusion,headSha,status
codex login status     # codex 인증 살아있는지
node /Users/yhkwon/.claude/plugins/cache/openai-codex/codex/1.0.2/scripts/codex-companion.mjs setup --json | grep reviewGateEnabled
```

확인 사항:
- main이 origin/main과 동기화돼있고 모든 CI ✅
- codex `Logged in using ChatGPT` (만료시 `! codex login` 요청)
- review-gate `true` (꺼져있으면 `--enable-review-gate`)

읽어야 할 문서 (순서 고정):
1. `CLAUDE.md` — 프로젝트 룰 (특히 "Post-Implementation Review" 섹션의 `/codex:review` 룰)
2. `docs/FEATURE_PARITY.md` — 우선순위 (B9-5는 close됨)
3. 작업 시작 전 해당 영역의 `docs/C_API_REFERENCE.md` 행 확인

---

## 작업 1 (✅ 완료, 2026-04-27): `jpeg_write_coefficients` 실구현

`crates/libjpeg-turbo-rs-capi/src/jpeglib.rs::jpeg_write_coefficients`가 stub에서 deferred-emit 본 구현으로 교체됨. 핸들 검증(magic), coding-mode 분기(progressive/arithmetic/optimize), 마커 placement(JFIF/APP14 후), 4-component CMYK/YCCK Adobe APP14, restart override 모두 적용. 통합 테스트 5개 (`capi_jpeglib_write_coefficients.rs`) 모두 그린.

⚠️ **남은 한계 — 작업 1.5 (transform path)에서 처리해야 함**.

---

## 작업 1.5: jpegtran transform path (`-rotate`/`-flip`/`-crop`)

### Goal
stock jpegtran의 -rotate/-flip/-transpose/-crop 옵션이 우리 cdylib에 link됐을 때 동작하도록 만든다. 작업 1은 read_coefficients → write_coefficients 단순 transcoding (`-copy all`, `-progressive`, `-arithmetic`, `-restart N`)까지 지원. transform path는 별개의 dependency.

### 현재 상태
우리 shim에 다음 함수가 없음 → jpegtran transupp.c가 호출하면 unresolved symbol:
- libjpeg memory manager: `jpeg_alloc_huff_table`, `alloc_barray`, `request_virt_barray`, `realize_virt_arrays`, `access_virt_barray` 등
- transupp helpers는 위 함수를 통해 destination cinfo에 virtual coefficient array를 alloc & 채움
- 그 결과 `jpeg_write_coefficients`로 전달되는 `coef_arrays`는 우리가 read_coefficients에서 stash한 `Box<CoefHandle>`이 아니라 stock libjpeg memory manager 결과 — 우리 magic 검증에서 reject됨 (의도적 안전 장치)

### 작업 분해
1. **memory manager API 구현** (~6-10h)
   - `j_common_ptr->mem` table (`jpeg_memory_mgr` struct) 채우기
   - `alloc_small`, `alloc_large`, `alloc_sarray`, `alloc_barray`, `request_virt_sarray`, `request_virt_barray`, `realize_virt_arrays`, `access_virt_sarray`, `access_virt_barray`, `free_pool`, `self_destruct`
   - 가상 array는 우리 사이드에서 simple in-RAM model로 충분 (libjpeg는 disk-backed temp file 옵션 있지만 jpegtran의 일반 케이스는 in-RAM)

2. **transupp.c가 의존하는 추가 entry points** (~2-4h)
   - `jpeg_get_small`/`jpeg_get_large`/`jpeg_open_backing_store`/`jpeg_open_backing_store_callback` 등
   - 우리 shim에서 stub 또는 in-RAM 구현

3. **`jpeg_write_coefficients` 측 수정** (~2-3h)
   - 현재: magic 매치 안 되면 reject
   - 추가: 우리 memory manager가 만든 virtual barray 핸들도 인식 (magic 별도 prefix or pointer table lookup)
   - virtual barray에서 coefficients 읽어 `JpegCoefficients` 형태로 materialize

4. **테스트** (~2h)
   - `examples/stock_djpeg_cjpeg/build.sh`에 jpegtran 추가
   - `cmp jpegtran -rotate 90 input.jpg vs upstream jpegtran -rotate 90 input.jpg` 픽셀(또는 byte) 비교

### Validation
```bash
bash examples/stock_djpeg_cjpeg/build.sh   # jpegtran 포함 빌드
$OUT_DIR/jpegtran -rotate 90 testimages/testorig.jpg > /tmp/our.jpg
upstream-jpegtran -rotate 90 testimages/testorig.jpg > /tmp/upstream.jpg
cmp /tmp/our.jpg /tmp/upstream.jpg   # byte-exact 목표
```

### 예상 크기
~12-19 시간 작업. memory manager가 가장 큰 부분.

### Risk / Pitfalls
- libjpeg memory manager는 pool 기반 — 우리 in-RAM 구현이 ABI-correct 해야 함
- virtual array의 access_*는 row offset 기반 lazy fetch — write/read 패턴 정확 매치 필요
- 12-bit, 16-bit precision 분기 (`jpeg12_write_coefficients`도 있음 — 같은 패턴)

---

## 작업 2 (추천 2순위 — 작은 win): SSE2 upsample width=2 kernel guard

### Goal
SSE2 upsample 커널이 width=2일 때 fancy 보간하는 vs 스칼라 box 동작 mismatch — pipeline 레이어에서 마스킹하고 있음. 커널 자체에 guard 넣어서 parity test 부활.

### 현재 상태
- 비활성화된 테스트: `tests/simd_x86.rs:116 sse2_upsample_edge_cases`
  - 라인 123 코멘트: "Two-sample case (width=2) is handled by the pipeline layer with a..."
- 커널: `src/simd/x86_64/upsample.rs` — `sse2_fancy_upsample_h2v1` (width=2 기준 SIMD 처리 주의 필요)
- 스칼라 reference: `src/decode/upsample.rs::fancy_upsample_h2v1`

### 작업 분해
1. **재현** (30m)
   - 비활성 테스트의 입력으로 SSE2 vs 스칼라 출력 차이 확인
   - 차이가 width=2 specific인지 width<8 일반인지 좁히기

2. **Kernel guard 추가** (1h)
   - `sse2_fancy_upsample_h2v1` 진입부에 `if input.len() < 8 { return scalar_fancy_upsample_h2v1(...) }` 패턴
   - libjpeg-turbo C 참조: `references/libjpeg-turbo/simd/x86_64/jdsample-sse2.asm` — width 분기 확인

3. **테스트 부활** (30m)
   - `tests/simd_x86.rs:sse2_upsample_edge_cases`의 `#[ignore]` 제거 (또는 부정 패턴 제거)
   - parity test가 SSE2 path를 강제로 타도록

### Validation
```bash
cargo test --test simd_x86 sse2_upsample_edge_cases
cargo test --test simd_parity   # 전체 SIMD parity 회귀
```

### 예상 크기
~30-50 lines. 2-3 시간.

### Risk
- AVX2 path도 같은 이슈 가능성 — 발견시 같이 fix

---

## 작업 3: B9-2 Pillow / B9-3 ImageMagick compat 활성화

### Goal
현재 `examples/pillow_smoke/` 와 `examples/imagemagick_smoke/`는 빌드/실행만 되고 통합 테스트는 없음 (또는 ignore). `tests/capi_pillow_compat.rs` / `tests/capi_imagemagick_compat.rs` 형태로 부활.

### 현재 상태
- `examples/pillow_smoke/run.sh` — Pillow venv + 모듈 빌드 + 우리 cdylib을 Pillow가 강제로 로드하게 symlink overlay 방식 (이미 설정됨)
- `examples/imagemagick_smoke/run.sh` — ImageMagick magick 바이너리 + 우리 cdylib 링크 (설정됨)
- 통합 테스트: 사전조사 — 위치 확인 필요

### 작업 분해
1. **사전조사** (1h)
   - `tests/capi_pillow_compat.rs` 존재 여부 + 현재 ignore 사유 확인
   - Pillow가 우리 shim을 어떤 함수까지 호출하는지 dlopen trace
   - 현재 어떤 단계에서 fail하는지 재현

2. **호환성 fix** (가변, 2-8h)
   - 누락된 jpeg_* symbol 채우기 (capi 크레이트에 추가)
   - Pillow가 expect하는 specific 동작과 우리 동작 차이 좁히기

3. **CI 통합** (1h)
   - `.github/workflows/ci.yml`에 Pillow / ImageMagick 단계 추가
   - 테스트의 `#[ignore]` 제거

### Validation
```bash
bash examples/pillow_smoke/run.sh   # 직접 실행
bash examples/imagemagick_smoke/run.sh
cargo test -p libjpeg-turbo-rs-capi --test capi_pillow_compat -- --include-ignored
```

### 예상 크기
가변 (4-15 시간). 발견되는 호환성 갭 수에 비례.

### Risk
- Python venv 환경 변수가 CI 마다 다름 (pillow wheel binary가 macOS arm64 vs linux x86_64에 다른 라이브러리 검색 경로)
- ImageMagick 버전별 ABI 차이

---

## 작업 4 (open-ended): 성능 추격 (experiments/)

### Goal
C libjpeg-turbo와의 성능 격차를 좁힘.

### 현재 상태
- `experiments/README.md`에 워크플로 정의됨
- 기존 TSV: `experiments/idct.tsv`, `experiments/huffman.tsv`, `experiments/color.tsv`, `experiments/upsample.tsv`, `experiments/pipeline.tsv`
- `examples/bench_c_decode_linux.c` — C 벤치마크 baseline

### 작업 분해 (1 사이클당)
1. **target 선정** — 어느 hot path?
   - 현재 hot paths (project memory 기준):
     - `crates/libjpeg-turbo-rs-capi/src/jpeglib.rs` (37x)
     - `src/decode/pipeline.rs` (25x)
     - `src/encode/pipeline.rs` (29x)

2. **profiling**
   ```bash
   sudo bash -c 'echo performance | tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor'
   sudo bash -c 'echo 1 > /sys/devices/system/cpu/intel_pstate/no_turbo'
   samply record cargo bench -- decode_640x480_420
   ```

3. **C SIMD 참조 학습** (CLAUDE.md 룰)
   - 해당 hot path의 C asm 읽고 알고리즘 이해
   - 참조 위치: `references/libjpeg-turbo/simd/x86_64/...` 또는 `references/libjpeg-turbo/simd/aarch64/...`

4. **One change at a time + 측정 + TSV 기록**

### Validation
```bash
# 변경 전후 양쪽 측정 (sequential, never parallel)
./bench_c_decode_linux > c_baseline.txt
cargo bench --bench decode > rust_after.txt
# experiments/<target>.tsv에 keep / discard / crash 기록
```

### 예상 크기
사이클당 2-8 시간. 무한 반복 가능.

### Risk
- CPU 주파수 안정화 안 한 측정은 noise로 fail
- 한 번에 두 가지 바꾸면 원인 분리 불가

---

## 추천 실행 순서

1. **작업 2 (SSE2 guard)** — 작고 명확. 워밍업.
2. **작업 1 (`jpeg_write_coefficients`)** — 큰 한 방. jpegtran lossless 완성.
3. **작업 3 (Pillow/ImageMagick)** — 외부 호환성, 발견 사항 기반 진행.
4. **작업 4 (성능)** — 다른 일 다 끝난 뒤 또는 별도 세션.

---

## Workflow 룰 (다음 세션에서도 적용)

- 각 commit 전: `cargo fmt --check` + `cargo clippy --lib -- -D warnings` + 해당 영역 `cargo test`
- **commit 후 push 전: `/codex:review --base HEAD~1`** (CLAUDE.md 룰 — 비-trivial 변경)
- push 후: `gh run watch <id> --exit-status > /tmp/x.log` + 별도로 `gh run view <id> --json conclusion` 검증 (exit code만 믿지 말 것)
- 큰 변경에 대해서는 작은 commit 여러 개로 분할
- TDD: 실패 테스트 → 최소 구현 → refactor 순서
- C 비교 검증 필수 (decode/encode/transform 모두): `examples/bench_c_decode_linux`, djpeg, cjpeg, jpegtran
