(a) Assume that every word is matched to an integer number k. Because the true empirical distribution $y​$ is a one-hot vector, for the word $o​$, only the entry in corresponding position in $\log(\hat{y}_o)​$ will have non-zero product.

$$-\sum\limits_{w\in Vocab}y_w\log(\hat{y}_w)= -\sum\limits_{w\in Vocab}\mathbb{1}\{w=o\}\log(\hat{y}_w)= -\log(\hat{y}_o)​$$

(b)

$$\begin{align}\nabla_{v_c}J(v_c, o, U) &= \dfrac{\partial}{\partial v_c}\left(-\log\dfrac{\exp(u_o^Tv_c)}{\sum_{w\in Vocab}\exp(u_w^Tv_c)}\right)\\ &= -\dfrac{\partial}{\partial v_c}\log\dfrac{\exp(u_o^Tv_c)}{\sum_{w\in Vocab}\exp(u_w^Tv_c)} \\&= -\dfrac{\partial}{\partial v_c}\left(\log\exp(u_o^Tv_c) - \log\sum_{w\in Vocab}\exp(u_w^Tv_c)\right) \\&= -\dfrac{\partial}{\partial v_c}\left(u_o^Tv_c - \log\sum_{w\in Vocab}\exp(u_w^Tv_c)\right)\\&= -\left(\dfrac{\partial u_o^Tv_c}{\partial v_c} - \dfrac{\partial}{\partial v_c}\log\sum_{w\in Vocab}\exp(u_w^Tv_c)\right)\\&= -\left(u_o - \sum_{w\in Vocab}\dfrac{\exp(u_w^Tv_c)}{\sum_{x\in Vocab}\exp(u_x^Tv_c)}u_w\right) \\&=-\left(u_o - \sum_{w\in Vocab}\hat{y}_wu_w\right)\\&=U\hat{y}-u_o = U\hat{y} - Uy = U(\hat{y} - y)\end{align}$$

(c)

$$\begin{align}\nabla_{u_w}J &=\dfrac{\partial}{\partial u_w}\left(-\log\dfrac{\exp(u_o^Tv_c)}{\sum_{x\in Vocab}\exp(u_x^Tv_c)}\right) \\&=-\dfrac{\partial}{\partial u_w}\log\dfrac{\exp(u_o^Tv_c)}{\sum_{x\in Vocab}\exp(u_x^Tv_c)} \\&=-\dfrac{\partial}{\partial u_w}\left(\log\exp(u_o^Tv_c) - \log\sum_{x\in Vocab}\exp(u_x^Tv_c)\right)\\&=\dfrac{\partial}{\partial u_w}\left(\log\sum_{x\in Vocab}\exp(u_x^Tv_c) - u_o^Tv_c\right)\\&=\dfrac{\partial}{\partial u_w}\log\sum_{x\in Vocab}\exp(u_x^Tv_c) - b \\&= \dfrac{\exp(u_w^Tv_c)}{\sum_{x\in Vocab}\exp(u_x^Tv_c)}v_c - b \\&= \hat{y}_wv_c-b \end{align}​$$

其中，当 $w= o$ 时，$b = v_c$，否则 $b = \mathbb{0}$

进一步：

$$\nabla_U J_{\text{naive-softmax}} = (\hat{y} - y)^T v_c$$

其中，在 $o$ 对应的位置，有 $v_c$，其余全部为 0

(d)

$$\nabla_x\sigma(x) = \dfrac{\text{d}}{\text{d}x}\dfrac{1}{1 + e^{-x}} = \dfrac{e^{-x}}{(1 + e^{-x})^2} = \sigma(x)(1- \sigma(x))​$$

(e) 

(i)

$$\begin{align}\nabla_{v_c} J_{\text{neg-sample}} &= -\dfrac{\partial}{\partial v_c}\left(\log(\sigma(u_o^Tv_c)) + \sum_{k=1}^K\log(\sigma(-u_k^Tv_c))\right)  \\&= -\left(\dfrac{\partial}{\partial v_c}\log(\sigma(u_o^Tv_c)) + \dfrac{\partial}{\partial v_c}\sum_{k=1}^K\log(\sigma(-u_k^Tv_c))\right) \\&= -\left(\dfrac{1}{\sigma(u_o^Tv_c)}\dfrac{\partial}{\partial v_c}\sigma(u_o^Tv_c) + \sum_{k=1}^K\dfrac{1}{\sigma(-u_k^Tv_c)}\dfrac{\partial}{\partial v_c} \sigma(-u_k^Tv_c)\right) \\&= -\left(\dfrac{\sigma(u_o^Tv_c)(1 - \sigma(u_o^Tv_c))u_o}{\sigma(u_o^Tv_c)} + \sum_{k=1}^K\dfrac{\sigma(-u_k^Tv_c)(1-\sigma(-u_k^Tv_c))(-u_k)}{\sigma(-u_k^Tv_c)}\right)\\&=- (1 - \sigma(u_o^Tv_c))u_o +  \sum_{k=1}^K(1-\sigma(-u_k^Tv_c))u_k\end{align}$$

(ii)

$$\begin{align}\nabla_{u_o} J_{\text{neg-sample}}&= - \dfrac{\partial}{\partial u_o}\log(\sigma(u_o^Tv_c))-\sum_{k=1}^K\dfrac{\partial}{\partial u_o}\log(\sigma(-u_k^Tv_c))  \\&= - \dfrac{\sigma(u_o^Tv_c)(1 -  \sigma(u_o^Tv_c))v_c}{\sigma(u_o^Tv_c)} \\&= -(1 -  \sigma(u_o^Tv_c))v_c\end{align}​$$

更进一步：

$$\nabla_U J_{\text{neg-sample}} = -v_c(1 - \sigma(U^Tv_c))^T$$

(iii)

$$\begin{align}\nabla_{u_k} J_{\text{neg-sample}} &= - \dfrac{\partial}{\partial u_k}\log(\sigma(u_o^Tv_c)) -\sum_{k=1}^K\dfrac{\partial}{\partial u_k}\log(\sigma(-u_k^Tv_c)) \\&=-\sum_{k=1}^K\dfrac{\partial}{\partial u_k}\log(\sigma(-u_k^Tv_c)) \\&=-\sum_{k=1}^K \dfrac{\sigma(-u_k^Tv_c)(1 - \sigma(-u_k^Tv_c))(-v_c)}{\sigma(-u_k^Tv_c)}\\&=-\sum_{k=1}^K (1 - \sigma(-u_k^Tv_c))(-v_c)\end{align}$$

<font color="red">Why this
loss function is much more efficient to compute than the naive-softmax loss?</font>

(f)

$$\begin{align}\dfrac{\partial}{\partial U}J_{\text{skip-gram}}(v_c,w_{t-m},\dots, w_{t+m}, U) &=\sum_{-m\le j\le m, j\ne 0} \dfrac{\partial}{\partial U}J(v_c, w_{t+j}, U)\end{align}$$

$$\begin{align}\dfrac{\partial}{\partial v_c}J_{\text{skip-gram}}(v_c,w_{t-m},\dots, w_{t+m}, U)=\sum_{-m\le j\le m, j\ne 0} \dfrac{\partial}{\partial v_c}J(v_c, w_{t+j}, U) \end{align}$$

$$\begin{align}\dfrac{\partial}{\partial v_w}J_{\text{skip-gram}}(v_c,w_{t-m},\dots, w_{t+m}, U)=\sum_{-m\le j\le m, j\ne 0} \dfrac{\partial}{\partial v_w}J(v_c, w_{t+j}, U) = 0\end{align} $$