# FLINT rerun on three more Dense differential cases

FLINT Hologuess won all three pairs on `Dense_r4d26` and `Dense_r26d4`.
Native Ore won all three on `Dense_r5d25`. The three cases follow
`Dense_r25d4` in the ordered list of Dense losses from the NumPy benchmark.
Times are medians in milliseconds. [Samples and medians](summary.json)

| Case | Prior NumPy Hologuess | Prior native Ore | FLINT Hologuess | Native Ore rerun |
|---|---:|---:|---:|---:|
| `Dense_r4d26` | 129.452 | 124.620 | 107.887 | 114.585 |
| `Dense_r26d4` | 123.758 | 115.870 | 109.568 | 114.722 |
| `Dense_r5d25` | 154.494 | 122.692 | 126.138 | 110.070 |

One $N=0.5$ s was declared for the differential class before the first
pair. Each pair started one fresh process per method with one native thread
and a common barrier after input and library setup. The measured interval
included public guessing and conversion to a canonical bounded operator
space. The controller then compared every coefficient with the independent
exact reference for the same fitting and held-out data.

All nine pairs completed and all 18 returned spaces matched the exact
references. Each case uses row zero. The fitting and held-out lengths are
158 and 8 for `Dense_r4d26`, 162 and 4 for `Dense_r26d4`, and 179
and 8 for `Dense_r5d25`. Input, source and bridge hashes matched and all
owned processes passed cleanup.

[Public corpus](../../README.md#upstream-compatibility-and-corpus-verification)
