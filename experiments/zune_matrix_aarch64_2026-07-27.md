# zune-jpeg wide-matrix decode baseline — aarch64 (issue #360)

Machine: Apple M-series (aarch64-apple-darwin), macOS, rustc 1.94.1, --release (lto). Ours = libjpeg_turbo_rs::decompress() @ main+#351 (0dcf778); zune = zune-jpeg 0.5.15 with default options. Metric = best-of-N (budgeted), single allocation profile per decoder. Harness: examples/bench_zune_matrix.rs. Date: 2026-07-27.

Run: cargo run --release --example bench_zune_matrix

```
case                         ours(us)   zune(us)  ratio   allocs        bytes  z.allocs      z.bytes  parity
gray_8x8                          3.1        2.4   1.32       12        27714        3          816  LEN-MISMATCH ours=64 zune=192
blue_16x16_420                    6.7        5.6   1.19       19        20070       18         4944  ok
blue_16x16_420_prog               9.1        8.0   1.14       51        51175       21         5488  ok
photo_64x64_420                  27.1       31.8   0.85       19        37542       18        24528  ok
photo_64x64_420_prog             62.3       68.5   0.91       51        80167       21        35920  ok
nonint_440_64x64                 10.8        8.1   1.33       17        47328       22        42320  ok
photo_320x240_420               412.5      556.8   0.74       19       365734       18       285648  ok
photo_320x240_422               516.9      709.4   0.73       17       556454       18       267088  ok
photo_320x240_444               777.7     1066.6   0.73       15       479654        5       247248  ok
photo_320x240_420_prog         1182.8     1278.7   0.93       51       626471       21       511568  ok
gray_227x149                     39.4       65.6   0.60        9        78625        3       105677  LEN-MISMATCH ours=33823 zune=101469
photo_640x480_420               906.7     1157.3   0.78       19      1403814       18      1030608  ok
photo_640x480_422              1788.2     2436.2   0.73       17      2169254       18       993488  ok
photo_640x480_444              2704.5     3672.8   0.74       15      1862054        5       953808  ok
photo_640x480_420_rst           413.1      558.0   0.74       19       365734       18       285648  ok
photo_640x480_422_prog         5007.8     5535.1   0.90       49      3428391       21      2213328  ok
photo_640x480_444_prog         7516.8     8186.6   0.92       47      3735591        8      2797008  ok
gradient_640x480                415.8      600.5   0.69       19      1403814       18      1030608  ok
graphic_640x480_420             350.7      410.3   0.85       19      1403814       18      1030608  ok
checker_640x480_420             823.5     1124.3   0.73       19      1403814       18      1030608  ok
photo_1280x720_420             4968.2     6680.5   0.74       19      4171174       18      2981328  ok
photo_1920x1080_420           11207.1    15063.5   0.74       19      9380774       18      6544848  ok
photo_1920x1080_422           13934.2    19286.8   0.72       17     14534054       18      6433488  ok
photo_1920x1080_444           21042.0    29058.8   0.72       15     12460454        5      6314448  ok
graphic_1920x1080_420          1752.8     1866.0   0.94       19      9380774       18      6544848  ok
photo_1920x1080_420_prog      33612.7    35130.1   0.96       51     15677991       21     12784848  ok
photo_1920x1080_422_prog      41405.8    43678.1   0.95       49     22858791       21     14701008  ok
photo_1920x1080_444_prog      61366.9    64080.5   0.96       47     24932391        8     18756048  ok
gray_900x675_prog              3307.7     3665.8   0.90       27      2514181        4      3066900  LEN-MISMATCH ours=607500 zune=1822500
photo_2560x1440_420           19878.1    26848.2   0.74       19     16617894       18     11490768  ok
photo_3840x2160_420           44881.6    60506.8   0.74       19     37359014       18     25529808  ok
photo_3840x2160_420_prog     134617.0   140301.1   0.96       51     62272551       21     50359248  ok
rw_2048x1536_q90               9222.2    11574.8   0.80       19     14182822       18      9782736  ok
rw_4k_420_q85                  8315.8     9586.5   0.87       19     37359014       18     25529808  ok
rw_4k_progressive             18610.0    17028.8   1.09       51     62272551       21     50359248  ok
rw_8k_420_q75                 28285.1    30309.3   0.93       19    149348774       18    100824528  ok
rw_8k_progressive             70683.8    53051.0   1.33       51    248911911       21    200249808  ok

summary: 29 wins / 5 losses / 0 ties (±2% threshold, 34 scored cases) + 3 format-mismatch cases (unscored)
losses: blue_16x16_420 (1.19x), blue_16x16_420_prog (1.14x), nonint_440_64x64 (1.33x), rw_4k_progressive (1.09x), rw_8k_progressive (1.33x)
note: LEN-MISMATCH cases compare different output formats (e.g. zune expands grayscale to RGB); their timings are printed for reference but excluded from win/loss.
```

## Reading

- 4:2:2 wins here (0.72-0.73) predate #350 on this arch — aarch64 never had the x86 4:2:2 regression; #350 removed the 4.2 MB full-res chroma traffic structurally.
- Remaining losses map to open issues: tiny-image tail (#351 follow-up territory), non-interleaved 4:4:0 (multi-scan path), and the progressive superlinear scaling (#352: 4K prog 1.10x -> 8K prog 1.33x, reproducing the EPYC pattern locally).
- Grayscale cases are LEN-MISMATCH by design: zune expands grayscale to RGB, we emit 1-channel — flagged, not scored.
