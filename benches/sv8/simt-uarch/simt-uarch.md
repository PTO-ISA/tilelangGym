# uarch

Load-4 on 128×128 issues `LDS`, `FMNMX`, `REDUX`, `FDIV`, `FMUL`, `FADD`, and `STS`. The programs here measure those ops on one 64×128 float tile. Every launch passed.

One issue is one warp of 32 lanes. Active cycles are the completion minus the issue of that one instruction. Elements per cycle is the issue count divided by the overlap-adjusted cycles, times 32. That count is the whole launch. One lane writes one element. It is not a per-thread rate.

`LDS`, `STS`, and `FADD` are the 32-bit programs from the earlier sweep. `FMUL`, `FDIV`, `FMNMX`, and `REDUX` are the new runs. The thread count is `threads` by 1.

## 1. LDS

One 32-bit issue moves 128 bytes, 4 bytes on each lane. The tile is 32768 bytes, so the sweep issues 256 loads. B/cycle is those bytes divided by the overlap-adjusted cycles.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">B/cycle</th>
</tr>
</thead>
<tbody>
<tr><td align="center">32</td><td align="center">4.74</td></tr>
<tr><td align="center">128</td><td align="center">27.07</td></tr>
<tr><td align="center">256</td><td align="center">35.61</td></tr>
<tr><td align="center">512</td><td align="center">105.07</td></tr>
<tr><td align="center">1024</td><td align="center"><b>114.32</b></td></tr>
<tr><td align="center">2048</td><td align="center">98.35</td></tr>
</tbody>
</table>

The active time is 27 cycles. The rate peaks at 1024 threads.

## 2. STS

The store writes the same 32768-byte tile, 128 bytes per issue.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">B/cycle</th>
</tr>
</thead>
<tbody>
<tr><td align="center">32</td><td align="center">8.09</td></tr>
<tr><td align="center">128</td><td align="center">34.60</td></tr>
<tr><td align="center">256</td><td align="center">59.95</td></tr>
<tr><td align="center">512</td><td align="center">118.03</td></tr>
<tr><td align="center">1024</td><td align="center"><b>123.12</b></td></tr>
<tr><td align="center">2048</td><td align="center">112.20</td></tr>
</tbody>
</table>

The active time is 20 cycles. The rate peaks at 1024 threads.

## 3. FMUL

`dst = src0 * src1`. One issue writes 32 output elements. The issue count stays 256. The active time stays 9 cycles.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Elements/cycle</th>
</tr>
</thead>
<tbody>
<tr><td align="center">32</td><td align="center">3.56</td></tr>
<tr><td align="center">128</td><td align="center">12.22</td></tr>
<tr><td align="center">256</td><td align="center">34.43</td></tr>
<tr><td align="center">512</td><td align="center">80.31</td></tr>
<tr><td align="center">1024</td><td align="center"><b>113.16</b></td></tr>
<tr><td align="center">2048</td><td align="center">103.40</td></tr>
</tbody>
</table>

## 4. FDIV

`dst = src0 / src1`. One issue writes 32 output elements. The issue count stays 256. The active time stays 18 cycles.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Elements/cycle</th>
</tr>
</thead>
<tbody>
<tr><td align="center">32</td><td align="center">1.78</td></tr>
<tr><td align="center">128</td><td align="center">11.00</td></tr>
<tr><td align="center">256</td><td align="center">23.85</td></tr>
<tr><td align="center">512</td><td align="center">41.55</td></tr>
<tr><td align="center">1024</td><td align="center">55.33</td></tr>
<tr><td align="center">2048</td><td align="center"><b>58.13</b></td></tr>
</tbody>
</table>

## 5. FMNMX

`dst = max(src0, src1)`. One issue writes 32 output elements. The issue count stays 256. The active time stays 7 cycles.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Elements/cycle</th>
</tr>
</thead>
<tbody>
<tr><td align="center">32</td><td align="center">4.57</td></tr>
<tr><td align="center">128</td><td align="center">15.11</td></tr>
<tr><td align="center">256</td><td align="center">44.60</td></tr>
<tr><td align="center">512</td><td align="center">100.71</td></tr>
<tr><td align="center">1024</td><td align="center">150.28</td></tr>
<tr><td align="center">2048</td><td align="center"><b>150.98</b></td></tr>
</tbody>
</table>

## 6. REDUX

`dst = reduce_max(src0)` across the warp. One issue reads 32 lanes and writes the max on every lane, 32 elements. The issue count stays 256. The active time stays 17 cycles.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Elements/cycle</th>
</tr>
</thead>
<tbody>
<tr><td align="center">32</td><td align="center">1.88</td></tr>
<tr><td align="center">128</td><td align="center">6.91</td></tr>
<tr><td align="center">256</td><td align="center">19.79</td></tr>
<tr><td align="center">512</td><td align="center">34.13</td></tr>
<tr><td align="center">1024</td><td align="center"><b>48.16</b></td></tr>
<tr><td align="center">2048</td><td align="center">40.98</td></tr>
</tbody>
</table>

## 7. FADD

`dst = src0 + src1`. One issue writes 32 output elements. The issue count stays 256. The active time stays 8 cycles.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Elements/cycle</th>
</tr>
</thead>
<tbody>
<tr><td align="center">32</td><td align="center">3.84</td></tr>
<tr><td align="center">128</td><td align="center">13.44</td></tr>
<tr><td align="center">256</td><td align="center">38.72</td></tr>
<tr><td align="center">512</td><td align="center">87.04</td></tr>
<tr><td align="center">1024</td><td align="center">126.72</td></tr>
<tr><td align="center">2048</td><td align="center"><b>127.04</b></td></tr>
</tbody>
</table>

## 8. Load-4

The shortest load-4 cell is 128×128 at 2048 threads. VF is 1887. `N` is 16384, so the kernel writes 8.68 output elements per cycle. Each of `LDS`, `FMNMX`, `REDUX`, `FDIV`, `FMUL`, `FADD`, and `STS` issues 512 times.

At 2048 threads the single-op rates are 103.40 for `FMUL`, 58.13 for `FDIV`, 150.98 for `FMNMX`, 40.98 for `REDUX`, and 127.04 for `FADD`. `REDUX` is the slowest of those five. The kernel's 8.68 output elements per cycle sits under that rate.
