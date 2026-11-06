# uarch

Each thread holds 32 independent float32 values, issued into registers before any use. The index is `tid + k * threads`, so one warp still touches 32 consecutive floats. Every launch is one block, `threads` by 1. Every launch passed.

One issue is one warp of 32 lanes. A 32-bit load or store moves 128 bytes. An add or a multiply-add writes 32 elements. The bytes or elements are divided by the cycles from the first issue of that opcode to its last completion. The active column is the median of completion minus issue. The element count is the whole launch. Divide by the thread count to get one thread.

1024 threads with two unified-buffer copies is 256 KB. The unified buffer is 248 KB, so the unified-buffer ops stop at 512 threads.

## 1. LDS

At 32 threads a warp holds 9 loads in flight. The rate is 19.60 bytes per cycle. A full 128 bytes per cycle would finish in one cycle per issue.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Issues</th>
<th align="center">Span</th>
<th align="center">Active</th>
<th align="center">B/cycle</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">32</td>
<td align="center">32</td>
<td align="center">209</td>
<td align="center">27</td>
<td align="center">19.60</td>
</tr>
<tr>
<td align="center">128</td>
<td align="center">128</td>
<td align="center">252</td>
<td align="center">27</td>
<td align="center">65.02</td>
</tr>
<tr>
<td align="center">256</td>
<td align="center">256</td>
<td align="center">394</td>
<td align="center">29</td>
<td align="center"><b>83.17</b></td>
</tr>
<tr>
<td align="center">512</td>
<td align="center">512</td>
<td align="center">1058</td>
<td align="center">36</td>
<td align="center">61.94</td>
</tr>
</tbody>
</table>

The load peaks at 256 threads, 83.17 bytes per cycle. At 512 threads each load stays active for 36 cycles and the rate falls.

SV1 sustains 114.98 bytes per cycle on `bw_b32_sep_ptr_e8192_t128_mi_off`. That run is 8192 floats and 128 threads, 32768 bytes over 285 cycles, and the load region contains only `LDS`. 85% of the 128 byte line is 108.8 bytes per cycle.

This program still issues an address add with each load. Those adds share the issue slots with `LDS`.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">LDS</th>
<th align="center">IADD in the burst</th>
<th align="center">STS in the burst</th>
<th align="center">B/cycle</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">128</td>
<td align="center">128</td>
<td align="center">149</td>
<td align="center">1</td>
<td align="center">65.02</td>
</tr>
<tr>
<td align="center">256</td>
<td align="center">256</td>
<td align="center">326</td>
<td align="center">15</td>
<td align="center">83.17</td>
</tr>
<tr>
<td align="center">512</td>
<td align="center">512</td>
<td align="center">894</td>
<td align="center">292</td>
<td align="center">61.94</td>
</tr>
</tbody>
</table>

At 512 threads the stores start before the loads finish, and the rate falls from the 256-thread peak.

## 2. LDG

The same 32 floats are read from global memory. The active time is 583 to 1146 cycles.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Issues</th>
<th align="center">Span</th>
<th align="center">Active</th>
<th align="center">B/cycle</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">32</td>
<td align="center">32</td>
<td align="center">2107</td>
<td align="center">1146</td>
<td align="center">1.94</td>
</tr>
<tr>
<td align="center">128</td>
<td align="center">128</td>
<td align="center">5722</td>
<td align="center">603</td>
<td align="center">2.86</td>
</tr>
<tr>
<td align="center">256</td>
<td align="center">256</td>
<td align="center">6341</td>
<td align="center">583</td>
<td align="center">5.17</td>
</tr>
<tr>
<td align="center">512</td>
<td align="center">512</td>
<td align="center">6276</td>
<td align="center">640</td>
<td align="center">10.44</td>
</tr>
<tr>
<td align="center">1024</td>
<td align="center">1024</td>
<td align="center">9918</td>
<td align="center">628</td>
<td align="center"><b>13.22</b></td>
</tr>
</tbody>
</table>

The global load peaks at 1024 threads, 13.22 bytes per cycle.

## 3. STS

The store writes 1.0 on each of the 32 floats. Active time stays 20 cycles.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Issues</th>
<th align="center">Span</th>
<th align="center">Active</th>
<th align="center">B/cycle</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">32</td>
<td align="center">32</td>
<td align="center">326</td>
<td align="center">20</td>
<td align="center">12.56</td>
</tr>
<tr>
<td align="center">256</td>
<td align="center">256</td>
<td align="center">497</td>
<td align="center">20</td>
<td align="center">65.93</td>
</tr>
<tr>
<td align="center">512</td>
<td align="center">512</td>
<td align="center">717</td>
<td align="center">20</td>
<td align="center"><b>91.40</b></td>
</tr>
</tbody>
</table>

The store peaks at 512 threads, 91.40 bytes per cycle.

## 4. FFMA

The program is `dst = a * b + a` after both inputs are loaded. The cells are output elements per cycle for the whole launch. Active time stays 11 cycles. The stores are issued in the same window as the arithmetic.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Issues</th>
<th align="center">Span</th>
<th align="center">Active</th>
<th align="center">Elements/cycle</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">32</td>
<td align="center">32</td>
<td align="center">548</td>
<td align="center">11</td>
<td align="center">1.87</td>
</tr>
<tr>
<td align="center">256</td>
<td align="center">256</td>
<td align="center">721</td>
<td align="center">11</td>
<td align="center"><b>11.36</b></td>
</tr>
<tr>
<td align="center">512</td>
<td align="center">512</td>
<td align="center">2223</td>
<td align="center">11</td>
<td align="center">7.37</td>
</tr>
</tbody>
</table>

The multiply-add peaks at 256 threads, 11.36 elements per cycle. One thread at that launch writes 11.36 / 256 = 0.044 elements per cycle.

## 5. FADD

The program is `dst = a + b` after both inputs are loaded. The cells are output elements per cycle for the whole launch. Active time stays 8 cycles. The stores are issued in the same window as the arithmetic.

<table align="center">
<thead>
<tr>
<th align="center">Threads</th>
<th align="center">Issues</th>
<th align="center">Span</th>
<th align="center">Active</th>
<th align="center">Elements/cycle</th>
</tr>
</thead>
<tbody>
<tr>
<td align="center">32</td>
<td align="center">32</td>
<td align="center">504</td>
<td align="center">8</td>
<td align="center">2.03</td>
</tr>
<tr>
<td align="center">256</td>
<td align="center">256</td>
<td align="center">896</td>
<td align="center">8</td>
<td align="center"><b>9.14</b></td>
</tr>
<tr>
<td align="center">512</td>
<td align="center">512</td>
<td align="center">2355</td>
<td align="center">8</td>
<td align="center">6.96</td>
</tr>
</tbody>
</table>

The add peaks at 256 threads, 9.14 elements per cycle. One thread at that launch writes 9.14 / 256 = 0.036 elements per cycle.
