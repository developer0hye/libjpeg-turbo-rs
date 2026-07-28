# zune-jpeg decode matrix — aarch64, 2026-07-28 (post-#376 calibration)

Command: `cargo run --release --example bench_zune_matrix`, run alone
(no parallel builds/tests), macOS aarch64 (Apple Silicon), quiet box.
This is the artifact behind README's "How it compares" row; it
supersedes `zune_matrix_aarch64_2026-07-27.md`, which predates the
#376 calibration fix (median-of-3 estimate + iteration floor + visible
medians) and whose per-case numbers are unreliable for the large
progressive cases.

Raw output:

```
   Compiling libjpeg-turbo-rs v0.8.0 (/Users/yhkwon/Documents/libjpeg-turbo-rs)
    Finished `release` profile [optimized] target(s) in 13.75s
     Running `/Volumes/T7/targets/libjpeg-turbo-rs/release/examples/bench_zune_matrix`
case                         ours(us)      o.med   zune(us)      z.med  ratio   allocs        bytes  z.allocs      z.bytes  parity
gray_8x8                          3.3        3.6        2.4        2.7   1.38       13        27746        3          816  LEN-MISMATCH ours=64 zune=192
blue_16x16_420                    6.8        7.3        5.6        6.2   1.20       19        20070       18         4944  ok
blue_16x16_420_prog               9.0        9.9        8.3        8.5   1.08       55        51253       21         5488  ok
photo_64x64_420                  27.3       29.4       32.2       34.8   0.85       19        37542       18        24528  ok
photo_64x64_420_prog             64.5       67.8       68.8       75.8   0.94       55        80335       21        35920  ok
nonint_440_64x64                 15.0       16.2        8.4        8.9   1.78       17        39264       22        42320  ok
photo_320x240_420               414.2      443.7      593.1      597.6   0.70       19       365734       18       285648  ok
photo_320x240_422               516.4      557.8      713.5      775.8   0.72       17       403494       18       267088  ok
photo_320x240_444               785.2      844.8     1087.5     1171.3   0.72       15       479654        5       247248  ok
photo_320x240_420_prog         1259.7     1324.9     1314.1     1371.2   0.96       55       628343       21       511568  ok
gray_227x149                     40.8       41.9       71.3       76.0   0.57        9        78625        3       105677  LEN-MISMATCH ours=33823 zune=101469
photo_640x480_420               942.5     1005.6     1188.5     1246.9   0.79       19      1403814       18      1030608  ok
photo_640x480_422              1862.6     1978.2     2516.6     2670.7   0.74       17      1556134       18       993488  ok
photo_640x480_444              2901.0     3246.5     4182.3     6096.6   0.69       15      1862054        5       953808  ok
photo_640x480_420_rst           459.9      726.2      559.1      563.1   0.82       19       365734       18       285648  ok
photo_640x480_422_prog         5150.3     5238.6     5534.0     5614.9   0.93       53      2824943       21      2213328  ok
photo_640x480_444_prog         7769.4     7891.3     8159.6     8281.0   0.95       51      3750063        8      2797008  ok
gradient_640x480                416.8      420.5      603.4      606.5   0.69       19      1403814       18      1030608  ok
graphic_640x480_420             348.3      352.9      412.5      417.6   0.84       19      1403814       18      1030608  ok
checker_640x480_420             824.0      864.1     1123.4     1143.3   0.73       19      1403814       18      1030608  ok
photo_1280x720_420             4983.3     5038.8     6683.3     6769.0   0.75       19      4171174       18      2981328  ok
photo_1920x1080_420           11221.5    11416.2    15106.8    15343.4   0.74       19      9380774       18      6544848  ok
photo_1920x1080_422           13917.8    14172.4    19482.5    19958.1   0.71       17     10390694       18      6433488  ok
photo_1920x1080_444           21108.0    21327.5    29364.9    29785.7   0.72       15     12460454        5      6314448  ok
graphic_1920x1080_420          1748.9     1775.0     1903.6     1930.3   0.92       19      9380774       18      6544848  ok
photo_1920x1080_420_prog      34255.9    34583.8    35278.2    35753.5   0.97       55     15727023       21     12784848  ok
photo_1920x1080_422_prog      42445.3    42833.4    43859.1    44260.4   0.97       53     18780303       21     14701008  ok
photo_1920x1080_444_prog      62578.3    63084.2    64213.0    65047.4   0.97       51     25029663        8     18756048  ok
gray_900x675_prog              3087.1     3140.5     3663.6     3715.5   0.84       29      2524009        4      3066900  LEN-MISMATCH ours=607500 zune=1822500
photo_2560x1440_420           20100.9    20277.0    27141.2    27431.5   0.74       19     16617894       18     11490768  ok
photo_3840x2160_420           45263.1    45604.1    61170.3    61664.5   0.74       19     37359014       18     25529808  ok
photo_3840x2160_420_prog     137639.9   138489.5   140899.8   141607.3   0.98       55     62467023       21     50359248  ok
rw_2048x1536_q90               9290.4     9466.6    11763.5    12346.8   0.79       19     14182822       18      9782736  ok
rw_4k_420_q85                  8414.9     8550.9     9675.6    10037.2   0.87       19     37359014       18     25529808  ok
rw_4k_progressive             11144.8    11388.1    17207.5    17515.1   0.65       55     62467023       21     50359248  ok
rw_8k_420_q75                 28330.2    28661.2    30902.9    31167.3   0.92       19    149348774       18    100824528  ok
rw_8k_progressive             41043.8    41854.4    53484.9    54391.8   0.77       55    249689583       21    200249808  ok

summary: 31 wins / 3 losses / 0 ties (±2% threshold, 34 scored cases) + 3 format-mismatch cases (unscored)
losses: blue_16x16_420 (1.20x), blue_16x16_420_prog (1.08x), nonint_440_64x64 (1.78x)
note: iteration floor (20) hit for: photo_3840x2160_420_prog — best-of-N for these rests on the minimum sample size; treat their ratios with care.
note: LEN-MISMATCH cases compare different output formats (e.g. zune expands grayscale to RGB); their timings are printed for reference but excluded from win/loss.
```
