# 最適化手法の数値実験例
人工的な関数に対して別資料で示した最適化アルゴリズムを適用する (ここではミニバッチなどは考慮していない)．

以下の3つの関数を用いる．

$$f_1(x, y) = \frac{1}{4}x^2 + y^2 \tag{1}$$$$ 
f_2(x, y) = x^3 + y^3 - 3xy \tag{2}$$$$
f_3(x, y) = -4 e^{-(x^2 + y^2)} - 2 e^{-((x + 4)^2 + (y + 4)^2)}  \tag{3}$$

それぞれに対して，

1. 最急降下法
1. Newton 法
1. Momentum 法
1. NAG
1. AdaGrad
1. RMSProp
1. Adam

を適用する．ただし，勾配は自動微分を用いて求めることにする．

<div style="page-break-before:always"></div>

### $f_1$ に対する最適化

<img src="https://github.com/SeeKT/ML_optimization/blob/master/note/fig/distorted/all/trajectory_distorted.gif?raw=true">

<img src="https://github.com/SeeKT/ML_optimization/blob/master/note/fig/distorted/all/value_distorted.svg?raw=true">

- この関数は凸関数．Newton 法が非常に高速．
- Momentum や NAG では値の振動が見られる．

<div style="page-break-before:always"></div>

### $f_2$ に対する最適化

<img src="https://github.com/SeeKT/ML_optimization/blob/master/note/fig/test_func_2/all/trajectory_test_func_2.gif?raw=true">

<img src="https://github.com/SeeKT/ML_optimization/blob/master/note/fig/test_func_2/all/value_test_func_2.svg?raw=true">

- Newton 法は2次の最適な方向に進むので，鞍点に陥っている．
- 簡単な関数形であるということもあり，1次の最適な方向に進み，学習率を Armijo rule で調整した steepest の収束が早い．

<div style="page-break-before:always"></div>

### $f_3$ に対する最適化

<img src="https://github.com/SeeKT/ML_optimization/blob/master/note/fig/test_func_3/all/trajectory_test_func_3.gif?raw=true">

<img src="https://github.com/SeeKT/ML_optimization/blob/master/note/fig/test_func_3/all/value_test_func_3.svg?raw=true">

- 2つの極小値以外の値がほぼ 0 であるような関数であり，通常の steepest の収束は非常に遅い．
- 一方で，学習率を適合的に調整する3つの手法の収束は早い．