# 最適化アルゴリズム (1)
ここでは，非線形最適化問題を解く代表的なアルゴリズムである最急降下法とニュートン法についてまとめる．[^1]を参考にした．

[^1]: 茨木, [最適化の数学](https://www.kyoritsu-pub.co.jp/kenpon/bookDetail/9784320015654), 共立出版,2011.

## 考える問題
以下のような制約なしの最適化問題を考える．

$$ \underset{\boldsymbol{x}}{\text{minimize}} \ \ f(\boldsymbol{x}) \ \ \text{subject to} \ \ \boldsymbol{x} \in S \subset \mathbb{R}^N.  \tag{1}$$

ここで，$f: \mathbb{R^N} \to \mathbb{R}$ とし，適当な微分可能性を仮定する．


## 解法
点 $\boldsymbol{x}^{*}$ の1次の最適性必要条件は停留点であること，つまり，

$$ \nabla f(\boldsymbol{x}^{*}) = 0 \tag{2}$$

である．非線形関数 $f$ に対して， (2) を直接解くのは一般に困難である．

$\rightsquigarrow$ 計算を繰り返すことで $\boldsymbol{x}^{*}$ に収束する点列 $\{ \boldsymbol{x}^k \}$ を生成し， $\boldsymbol{x}^{*}$ に十分近くなったと判断されたところで計算を終え，その時点の $\boldsymbol{x}^k$ を $\boldsymbol{x}^{*}$ の近似解として出力するのが普通．

現在の点 $\boldsymbol{x}^k$ において，ベクトル $d(\boldsymbol{x}^k) \in \mathbb{R}^N$ が

$$ \nabla f(\boldsymbol{x}^k)^{\mathrm{T}} d(\boldsymbol{x}^k) < 0 \tag{3}$$

を満たすならば， $\nabla f(\boldsymbol{x}^k)$ ($f(\boldsymbol{x}^k)$ からの最急増加方向) と $d(\boldsymbol{x}^k)$ は鈍角をなす．

$\rightsquigarrow$ $d(\boldsymbol{x}^k)$ は $f$ の降下方向を示す．

よって，降下方向ベクトル $d(\boldsymbol{x}^k)$ を見つけ，

$$ \boldsymbol{x}^{k + 1} = \boldsymbol{x}^k + \varepsilon_k d(\boldsymbol{x}^k) \tag{4}$$

という修正を行うという方針をとる．$\varepsilon_k$ はステップ幅または**学習率**と呼ばれる実数である．

$d(\boldsymbol{x}^k)$ や $\varepsilon_k$ をどのように見つけるかによってアルゴリズムが決まる．


## 最急降下法 (Steepest descent / Gradient descent)
$d(\boldsymbol{x}^k)$ として，勾配ベクトル $\nabla f(\boldsymbol{x}^k)$ の反対方向

$$ d(\boldsymbol{x}^k) = - \nabla f(\boldsymbol{x}^k) \tag{5}$$

を用いるアルゴリズム．1次の方法である．

- $d(\boldsymbol{x}^k)$ は $\boldsymbol{x}^k$ から見ると最急降下方向．

$f(\boldsymbol{x})$ の関数形によっては更新則 (5) の方向に進んでも関数値が増加することもある．

$\rightsquigarrow$ 学習率 $\varepsilon_k$ を

$$ \min \{f(\boldsymbol{x}^k + \varepsilon d(\boldsymbol{x}^k)) \, | \, \varepsilon > 0\} \tag{6}$$

を実現する $\varepsilon$ が存在すればその値に設定する (直線探索)．

最急降下法のアルゴリズムを以下に示す．

### Algorithm (Steepest)
#### Input
- $f$: 目的関数
- $\nabla f$: 目的関数の勾配
- $\delta$: 閾値
#### Output
- $\boldsymbol{x}^{*}$ の近似解 $\boldsymbol{x}^k$

#### 動作
1. (初期化) 初期値 $\boldsymbol{x}^0$ を定める．$k \leftarrow 0$.
1. (終了判定) $\|f(\boldsymbol{x}^{k + 1}) - f(\boldsymbol{x}^k)  \| < \delta$ であれば，$\boldsymbol{x}^k$ を出力して終了．
1. (反復) $d(\boldsymbol{x}^k) \leftarrow - \nabla f(\boldsymbol{x}^k)$. $\varepsilon_k \leftarrow$ 直線探索の近似解とし，(4) で $\boldsymbol{x}^{k + 1}$ を求める．$k \leftarrow k + 1$ として 2. へ戻る．

### Remark
- 最急降下法は，1次近似で最良の方向へ進む．
- $\varepsilon_k$ が十分に小さいとき，以下の ODE で表される (修正方程式)．
    $$ \dot{\boldsymbol{x}} = -\nabla f(\boldsymbol{x}) $$
- この方法では，勾配はすべてのデータを用いることで求められる．

### 線形探索
線形探索として Armijo のルールを用いる．Armijo のルールは以下のような更新則である．

$\beta, \gamma \in (0, \ 1)$ を選び，

$$ f(\boldsymbol{x}^k + \beta^{\ell_k} d(\boldsymbol{x}^k)) - f(\boldsymbol{x}^k) \leq \gamma \beta^{\ell_k} \nabla f(\boldsymbol{x}^k)^{\mathrm{T}} d(\boldsymbol{x}^k) \tag{7}$$

を満たす最小の非負整数 $\ell_k$ を求め，$t_k = \beta^{\ell_k}$ とし，この $t_k$ を学習率 $\varepsilon_k$ とする．

### ニュートン法
$\varepsilon = 1$ として $d(\boldsymbol{x}^k)$ として

$$ d(\boldsymbol{x}^k) = - \nabla^2 f(\boldsymbol{x}^k)^{-1} \nabla f(\boldsymbol{x}^k) \tag{8}$$

を選ぶ方法．$\nabla^2 f(\boldsymbol{x}^k)$ は $f$ の $\boldsymbol{x} = \boldsymbol{x}^k$ におけるヘッセ行列．正則性は仮定している．

以下，アルゴリズムの形に記述する．

### Algorithm (Newton)
#### Input
- $f$: 目的関数
- $\varepsilon$: 閾値
#### Output
- $\boldsymbol{x}^{*}$ の近似解 $\boldsymbol{x}^k$

#### 動作
1. (初期化) 初期値 $\boldsymbol{x}^0$ を定める．$k \leftarrow 0$.
1. (終了判定) $\|f(\boldsymbol{x}^{k + 1}) - f(\boldsymbol{x}^k)  \| < \varepsilon$ であれば，$\boldsymbol{x}^k$ を出力して終了．
1. (反復) $d(\boldsymbol{x}^k) \leftarrow - \nabla^2 f(\boldsymbol{x}^k)^{-1} \nabla f(\boldsymbol{x}^k)$, $\varepsilon_k \leftarrow 1$ とし，(4) で $\boldsymbol{x}^{k + 1}$ を求める．$k \leftarrow k + 1$ として 2. へ戻る．

#### Remark
ニュートン法は2次の最適な方向へ進む．実際，Taylor 展開を考えると

$$ f(\boldsymbol{x}^k + \boldsymbol{d}) \approx f(\boldsymbol{x}^k) + \nabla f(\boldsymbol{x}^k)^{\mathrm{T}} \boldsymbol{d} + \frac{1}{2} \boldsymbol{d}^{\mathrm{T}} \nabla^2 f(\boldsymbol{x}^k) \boldsymbol{d} \tag{9}$$

を得る．$\boldsymbol{d}$ を変数ベクトルと見て，この関数の停留点条件を求めると，

$$ \nabla f(\boldsymbol{x}^k) + \nabla^2 f(\boldsymbol{x}^k) \boldsymbol{d} = 0,$$

つまり，(8)を得る．

ニュートン法は収束が非常に早いが，高次元になると計算が困難になるので，実用的ではない．