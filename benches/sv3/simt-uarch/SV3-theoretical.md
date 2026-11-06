# SV3 theoretical

This is the SIMT GEMV at 256 threads. `A` and `x` are fp16. `Out` is fp32. `x` is loaded once.

The bytes read are `2 × N × K + 2 × K`. The bytes written are `4 × N`.

The load line is 128 bytes per cycle, one `LDS.b32`. The case is expected to reach at least 85% of that line. 85% of 128 is 108.8 bytes per cycle. SV1 sustained 114.98 on the same line, which is 90%.

```text
line     = (2 × N × K + 2 × K) / 128
expected = (2 × N × K + 2 × K) / 108.8
```

`expected` is the cycle window the case should finish inside. A measured window at or below `expected` has reached the 85% bar.

**Table 1.** Bytes, the 128 B/cycle line, and the 85% window.

<table align="center">
<thead>
<tr>
<th align="center">Shape</th>
<th align="center">Bytes</th>
<th align="center">At 128</th>
<th align="center">Expected, 85%</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">N=4096, K=8</td>
<td align="center">65552</td>
<td align="center">512.1</td>
<td align="center"><b>602.5</b></td>
</tr>
<tr>
<td align="center">N=4096, K=16</td>
<td align="center">131104</td>
<td align="center">1024.3</td>
<td align="center"><b>1205.0</b></td>
</tr>
<tr>
<td align="center">N=4096, K=32</td>
<td align="center">262208</td>
<td align="center">2048.5</td>
<td align="center"><b>2410.0</b></td>
</tr>
<tr>
<td align="center">N=4096, K=64</td>
<td align="center">524416</td>
<td align="center">4097.0</td>
<td align="center"><b>4820.0</b></td>
</tr>
<tr>
<td align="center">N=512, K=128</td>
<td align="center">131328</td>
<td align="center">1026.0</td>
<td align="center"><b>1207.1</b></td>
</tr>
<tr>
<td align="center">N=1024, K=128</td>
<td align="center">262400</td>
<td align="center">2050.0</td>
<td align="center"><b>2411.8</b></td>
</tr>
<tr>
<td align="center">N=2048, K=128</td>
<td align="center">524544</td>
<td align="center">4098.0</td>
<td align="center"><b>4821.2</b></td>
</tr>
</tbody>
</table>

**Table 2.** The 85% window against the measured buffer SIMT VF. The ratio is the measured VF divided by the expected window. A ratio of 1.00 has hit the bar. Above 1.00 the measured window is longer.

<table align="center">
<thead>
<tr>
<th align="center">Shape</th>
<th align="center">Expected</th>
<th align="center">Actual VF</th>
<th align="center">Actual / expected</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">N=4096, K=8</td>
<td align="center">602.5</td>
<td align="center">9189</td>
<td align="center">15.25</td>
</tr>
<tr>
<td align="center">N=4096, K=16</td>
<td align="center">1205.0</td>
<td align="center">33802</td>
<td align="center">28.05</td>
</tr>
<tr>
<td align="center">N=4096, K=32</td>
<td align="center">2410.0</td>
<td align="center">132303</td>
<td align="center"><b>54.90</b></td>
</tr>
<tr>
<td align="center">N=4096, K=64</td>
<td align="center">4820.0</td>
<td align="center">265690</td>
<td align="center">55.12</td>
</tr>
<tr>
<td align="center">N=512, K=128</td>
<td align="center">1207.1</td>
<td align="center">66724</td>
<td align="center">55.28</td>
</tr>
<tr>
<td align="center">N=1024, K=128</td>
<td align="center">2411.8</td>
<td align="center">133802</td>
<td align="center">55.48</td>
</tr>
<tr>
<td align="center">N=2048, K=128</td>
<td align="center">4821.2</td>
<td align="center">267071</td>
<td align="center">55.40</td>
</tr>
</tbody>
</table>

At `N = 4096`, `K = 32` the bar is 2410 cycles. The buffer VF is 132303 cycles, 54.90 times that window. The same shape at the full 128 B/cycle line is 2048.5 cycles.

On that buffer run the `LDS` window is the VF. Two launches, 2304 loads each. The next load is issued on a median gap of 32 cycles. The 262208 bytes over the issue span of 130398 cycles are 2.01 bytes per cycle. The first issue to the last completion is 131641 cycles, and the VF is 132303.

**Table 3.** Buffer `LDS` against the VF. `N = 4096`, `K = 32`, 256 threads.

<table align="center">
<thead>
<tr>
<th align="center"></th>
<th align="center">Launch 0</th>
<th align="center">Launch 1</th>
<th align="center">Sum</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">VF</td>
<td align="center">66143</td>
<td align="center">66160</td>
<td align="center">132303</td>
</tr>
<tr>
<td align="center">LDS issues</td>
<td align="center">2304</td>
<td align="center">2304</td>
<td align="center">4608</td>
</tr>
<tr>
<td align="center">Issue span</td>
<td align="center">65248</td>
<td align="center">65150</td>
<td align="center">130398</td>
</tr>
<tr>
<td align="center">First issue to last done</td>
<td align="center">65821</td>
<td align="center">65820</td>
<td align="center"><b>131641</b></td>
</tr>
<tr>
<td align="center">Head, before the first load</td>
<td align="center">90</td>
<td align="center">108</td>
<td align="center">198</td>
</tr>
<tr>
<td align="center">Tail, after the last load</td>
<td align="center">232</td>
<td align="center">232</td>
<td align="center">464</td>
</tr>
</tbody>
</table>

The other 662 cycles are that head and that tail. The head is address math: `S2R`, `SHFI`, `IADD`, `IMAD`, and `LEA` build the pointer. The tail is the last `F2F` and `FFMA`, a block barrier, the store addresses, and 32 `STS`. The minimum active time of a load is 27 and 30 cycles. The median active time is 652, which is queue time behind loads already issued.
