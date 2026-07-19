```
Algorithm 1: MCP-Net Forward Pass
Input: image I ∈ R^(128×128×1), num_scales S=5
Output: segmentation map ŷ ∈ R^(128×128×4)

1:  P₀ ← InputPyramid(I, S)          ▷ [I, pool(I), pool²(I), pool³(I), pool⁴(I)]
2:  conx ← ∅
3:  for depth d = 1 to S do
4:      D_d ← []
5:      for each scale i in remaining scales at depth d do
6:          if conx ≠ ∅ then
7:              conx ← MaxPool(conx)
8:              c ← Concat(conx, P_{d-1}[i])
9:          else
10:             c ← P_{d-1}[i]
11:         f ← ActiveConv2D(c, filters_d, kernel=3)   ▷ Conv→ReLU→BatchNorm
12:         D_d.append(f)
13:     conx ← D_d[0]                 ▷ finest scale at this depth feeds forward
14:     remaining scales ← remaining scales minus one (drop coarsest tracked scale)
15: bottleneck ← D_S[0]               ▷ single scale remains at depth S

16: x ← bottleneck
17: for depth d = S-1 down to 1 do
18:     u ← UpsampleConv(x)           ▷ nearest-neighbor upsample + conv
19:     u ← Concat(u, D_d[0])         ▷ skip connection from matching depth
20:     x ← ActiveConv2D(u, filters_d, kernel=3)
21:     x ← ActiveConv2D(x, filters_d, kernel=3)

22: ŷ ← Softmax(Conv1×1(x, filters=4))
23: return ŷ
```

````
Algorithm 2: Training with Snapshot Ensembling
Input: training set D_train, cycles C=10, epochs E=300, max_lr=0.01
Output: snapshot set M = {m_1, ..., m_C}

1:  epochs_per_cycle ← E / C
2:  for epoch e = 0 to E-1 do
3:      lr ← (max_lr/2) · (cos(π · (e mod epochs_per_cycle) / epochs_per_cycle) + 1)
4:      set optimizer learning rate to lr
5:      train one epoch on D_train using weighted Dice loss
6:          (weights: 0.36 RV, 0.34 Myo, 0.29 LV, 0.01 background)
7:      if (e+1) mod epochs_per_cycle == 0 then
8:          save current model weights as snapshot m_{(e+1)/epochs_per_cycle}
9:  return M

Algorithm 3: Ensemble Inference
Input: snapshot set M, input slice x
Output: predicted label map ŷ

1:  for each model m_i in M do
2:      p_i ← softmax_output(m_i, x)
3:  p̄ ← mean(p_1, ..., p_|M|)          ▷ soft-vote across snapshots
4:  ŷ ← argmax(p̄)
5:  return ŷ
````