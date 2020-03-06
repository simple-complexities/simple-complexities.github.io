---
layout: post
title:  "Functional Gradient Descent"
date:   2020-03-04 10:00:00 +0530
categories: optimization functional gradient descent
---

The content in this post has been adapted from [Functional Gradient Descent - Part 1](http://www.cs.cmu.edu/~16831-f12/notes/F12/16831_lecture21_danielsm.pdf) and [Part 2](http://www.cs.cmu.edu/~16831-f14/notes/F10/16831_lecture24_varunnr/16831_lectureNov11.vramakri.pdf).
Functional Gradient Descent was introduced in the NIPS publication [Boosting Algorithms as Gradient Descent](https://papers.nips.cc/paper/1766-boosting-algorithms-as-gradient-descent.pdf) by Llew Mason, Jonathan Baxter, Peter Bartlett and Marcus Frean in the year 2000.

We are all familiar with gradient descent for linear functions $$f(x) = w^Tx$$.  
Once we define a loss $$L$$, gradient descent does the following update steps ($$\alpha$$ is a parameter called the learning rate.):
\\[
    w \rightarrow w - \alpha \nabla L(w)
\\]
where we move around in the space of weights.
An example of a loss $$L$$ is:
\\[
    L(w) = \sum_{i=1}^n(y_i - w^Tx_i)^2 + \lambda\lVert w \rVert ^2
\\]
where the first term (the 'L2' term) measures how close $$f(x)$$ is close to $$y$$, while the second term (the 'regularization' term) accounts for the 'complexity' of the learned function $$f$$.

Suppose we wanted to extend $$L$$ to beyond linear functions $$f$$. We want to minimize something like:
\\[
    L(f) = \sum_{i=1}^n(y_i - f(x_i))^2 + \lambda\lVert f \rVert ^2
\\]
where $$\lVert f \rVert ^2$$ again serves as a regularization term, and we have updates of the form:
\\[
    f \rightarrow f - \alpha \nabla L(f)
\\]
where we move around in the space of functions, not weights!

Turns out, this is completely possible! And goes by the name of 'functional' gradient descent, or gradient descent in function space. But what does this mean?


### Functionals
A functional is a function defined over functions, returning a real value.  
**Examples:**
* The evaluation functional
\\[
    E_x(f) = f(x)
\\]
* The sum functional
\\[
    S_{\{x_1, \ldots, x_n\}}(f) = \sum_{i = 1}^n f(x_i) dx
\\]
* The integration functional
\\[
    I_{[a, b]}(f) = \int_a^b f(x) dx
\\]

It follows that the composition of a function $$g: \mathbb{R} \to \mathbb{R}$$ with a functional is also a functional.
The loss function $$L(f)$$ defined above is a functional!

### Reproducing Kernel Hilbert Space
It turns out that it is especially convenient when functions come from a special set, called a reproducing kernel Hilbert space.
A Hilbert space can be thought of as a vector space, where we have the notion of an inner product between two elements. (This is not the complete definition, but this is what we'll need).

---

#### Review: Kernels
A kernel $$K: X \times X \to \mathbb{R}$$ is a function that generalizes dot products:
* *Symmetry*: For any $$x_i, x_j \in X$$:
\\[
    K(x_i, x_j) = K(x_j, x_i).
\\]
* *Positive Semi-Definiteness*: For any $$x_1, \ldots, x_n \in X$$, the matrix $$K_M$$ given by 
\\[
    K_{M_{ij}} = K(x_i, x_j)
\\]
is positive semi-definite. Note that this implies $$K(x_i, x_j) \geq 0$$ always.

It turns out (Mercer's condition) that these conditions are equivalent to
\\[
    K(x_i, x_j) = \phi(x_i) \cdot \phi(x_j)
\\]
where $$\phi$$ is a function that is sometimes called the 'feature map'. Thus, a kernel can be thought of as the dot product in some feature space (the range of $$\phi$$).
Similar to the dot product then, the kernel measures similarity between two inputs.

Examples of kernel functions include:
* Linear Kernel: 
\\[
    K(x_i, x_j) = x_i \cdot x_j 
\\]
* Polynomial Kernel (of degree $$d$$): 
\\[
    K(x_i, x_j) = (x_i \cdot x_j + c)^d 
\\]
The presence of the $$c$$ term allows coefficients of degree less than $$d$$ to be accommodated too.  
* RBF Kernel (of 'width' $$\sigma$$): 
\\[
    K(x_i, x_j) = \exp\left(-\frac{\lVert x_i - x_j \rVert^2}{2\sigma^2}\right)
\\]

Try to derive what the associated feature map is, for each of these kernels!

---
&nbsp;

We can now define a reproducing kernel Hilbert space or a 'RKHS'.  
A **reproducing kernel Hilbert space**, obtained on fixing a kernel $$K$$, is a space of functions where every function $$f$$ is some linear combination of the kernel $$K$$ evaluated at some 'centers' $$x_C$$:
\\[
    f(x) = \sum_{i = 1}^n \alpha_i K(x, x_{Ci}) 
\\]
or, ignoring the argument $$x$$:
\\[
    f = \sum_{i = 1}^n \alpha_i K(\cdot, x_{Ci}) 
\\]
For a kernel $$K$$ will denote the associated reproducing kernel Hilbert space by $$H_K$$.  
From the definition above, every $$f \in H_K$$ is completely determined by the coefficients $$\alpha_f$$ and the centers $$x_{Cf}$$. Note that the number of centers ($$ = $$ dimension of $$\alpha_f$$) can vary between functions.

We can now define the inner product (the 'dot' product) in $$H_K$$ by:
\\[
    f \cdot g =  \sum_{i = 1}^{n_f} \sum_{j = 1}^{n_g} \alpha_{f_i} \alpha_{g_j} K(x_{Cf_i}, x_{Cg_j}) = \alpha_f K_{fg} \alpha_g
\\]
where,
\\[
    K_{\{fg}\_{ij}} = K(x_{Cf_i}, x_{Cg_j}).
\\]
This inner product induces the norm $$\lVert \cdot \rVert$$:
\\[
    {\lVert f \rVert}^2 = f \cdot f = \alpha_f K_{ff} \alpha_f \geq 0.
\\]
Why do we use the term reproducing? This is because we can 'reproduce' the value of $$f \in H_K$$ at any $$x$$ by taking the inner product of $$f$$ with the 'reproducing kernel' function $$ K(x, \cdot) \in H_K$$:
\\[
    f \cdot K(x, \cdot) = f(x).
\\]
Verify this property!

So, we've seen how to define the inner-product and norm $$\lVert f \rVert$$ of any function $$f \in H_K$$. But, in order to minimize via gradient descent, we need the definition of a derivative.

### Derivatives of Functionals
As reviewed in my previous post on [optimization theory](/optimization/constrained/theory/2020/03/03/optimization-review.html), one of the definitions of the derivative $$Df$$ of a function $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ is:
\\[
    \lim_{\\lVert h \rVert \rightarrow 0} \frac{\lVert{f(x + h) - (f(x) + Df(x) \cdot h)}\rVert}{\lVert{h}\rVert} = 0
\\]
where $$Df(x)$$ is a size $$n$$ row vector, with which the take the dot product of the direction $$h$$ with.

We may not be working in $$\mathbb{R}^n$$ anymore, but this definition gives us that the derivative $$DE$$ of a functional $$E$$ on $$H_K$$ must satisfy:
\\[
    \lim_{\\lVert h \rVert \rightarrow 0} \frac{\lVert{E(x + h) - (E(x) + DE(x) \cdot h)}\rVert}{\lVert{h}\rVert} = 0
\\]
where $$h$$ and $$x$$ are now functions in $$H_K$$, instead of points. This means that $$DE(x)$$ is a function, too! (Recall that addition of functions occurs point-wise.)

---
**Example 1:**
Let us take the example of the evaluation functional $$E_x(f) = f(x)$$ and compute its derivative:
\\[
    \begin{aligned}
        E(f + h) &= (f + h)(x)  \newline
                 &= f(x) + h(x) \newline
                 &= E(f) + h(x) \newline
                 &= E(f) + K(x, \cdot) \cdot h \newline
    \end{aligned}
\\]
Thus, the derivative $$DE_x(f)$$ is independent of $$f$$, and is given by:
\\[
        DE_x(f) = K(x, \cdot).
\\]

**Example 2:**
Similarly, following the example of my reference material, the functional $$E(f) = {\lVert f \rVert}^2$$ satisfies:
\\[
    \begin{aligned}
        E(f + h) &= {\lVert f + h \rVert}^2  \newline
                 &= {(f + h) \cdot (f + h)} \newline
                 &= {f \cdot f} + 2 {f \cdot h} + {h \cdot h} \newline
                 &= E(f) + 2 {f \cdot h} + {h \cdot h} \newline
    \end{aligned}
\\]
Thus, the derivative $$DE(f)$$ is defined as:
\\[
        DE(f) = 2f.
\\]
Note how similar this is to the derivative $$2x$$ of the function $$x \to \lvert x \rvert^2$$ on real numbers!

---
&nbsp;

#### The Chain Rule
Very fortunately, we also have the chain rule!
As discussed before, if $$E$$ is a functional and $$g: \mathbb{R} \to \mathbb{R}$$ is differentiable, then $$g(E)$$ is also a functional, with derivative:
\\[
    D(g(E))(f) = g'(E(f)) DE(f).
\\]

---

**Example 3:**
Let us compute the derivative of the loss functional $$L(f) = \sum_{i=1}^n(y_i - f(x_i))^2 + \lambda\lVert f \rVert ^2 $$ with the chain rule:
The individual terms in the first sum term is a composition of
\\[
    g_i(x) = (y_i - x)^2 \text{ and } E_{x_i}.
\\]
Thus, each of these terms has derivative:
\\[
    D(g_i({x_i}))(f) = -2 (y_i - E_{x_i}(f)) \cdot DE_{x_i}(f) = -2 (y_i - f(x_i)) \cdot K(x_i, \cdot).
\\]
The second term has derivative $$2f$$, as derived above.
Thus, the derivative $$DL(f)$$ is given by:
\\[
    DL(f) = \sum_{i = 1}^n -2 (y_i - f(x_i)) \cdot K(x_i, \cdot) + 2f.
\\]

---
&nbsp;

There is one last point to note. When we take steps in 'ordinary' gradient descent, we move along the negative of the gradient vector because that is the direction along which the dot product with the gradient is minimum. No matter how this vector points! In some sense, we are not restricted to move in any direction. This is because our underlying domain is $$\mathbb{R}^n$$.

In 'functional' gradient descent, however, we are restricted to $$H_K$$. How can we guarantee that when moving along $$DL(f)$$, we do not stray out of $$H_K$$? One way to ensure that is by proving that we always have $$DL(f) \in H_K$$. (This was true for our examples above! We have actually implicitly assumed this in our definition, too.) Then, closure of $$H_K$$ under addition (and scalar multiplication) ensures that at every iteration, our current function $$f$$ is in $$H_K$$.

This is what we will prove, in the next section.

The [Boosting Algorithms as Gradient Descent](https://papers.nips.cc/paper/1766-boosting-algorithms-as-gradient-descent.pdf) paper above, does not use reproducing kernel Hilbert spaces, and actually applies to more general sets of functions. This is why they mention the fact that moving along the gradient is not always possible. Instead, they move along the direction with the least dot product with the gradient, among all directions that keeps them within their domain of functions. With reproducing kernel Hilbert spaces, this is not a problem: we are, fortunately, not restricted to move along the negative of the gradient.

### $$H_K$$ is Closed under the Derivative
If $$E$$ is a functional on $$H_K$$, and $$f \in H_K$$, then we always have:
\\[
    DE(f) \in H_K.
\\]

Let us define the derivative $$DE^*(f)$$ as a functional:
\\[
    \lim_{\\lVert h \rVert \rightarrow 0} \frac{\lVert{E(f + h) - (E(f) + DE^\*(f)(h))}\rVert}{\lVert{h}\rVert} = 0
\\]
I differentiate between $$DE^*(f)$$ (the functional) and $$DE(f)$$ (the function).
We want to show that, in fact:
\\[
    DE^\*(f)(h) = {\langle h, DE(f) \rangle}_K
\\]

Note that $$DE^*(f)$$ is a linear functional!
Why? Using the definition above (and properties of the norm and limits):
* $$DE^*(f)(ch) = c \cdot DE^*(f)(h)$$ where $$c \in \mathbb{R}$$.
* $$DE^*(f)(h + g) = DE^*(f)(h) + DE^*(f)(g)$$.  

This should not be surprising! We use $$DE^*(f)$$ to give us the 'best' linear approximation around $$f$$ along each direction $$h$$.

The [Riesz Representation Theorem](https://en.wikipedia.org/wiki/Riesz_representation_theorem) then tells us that every linear functional $$L$$ on a Hilbert space is actually of the form:
\\[
    L = \langle \cdot, v \rangle
\\]
for some $$v$$ in the Hilbert space, where $$\langle \cdot, \cdot \rangle$$ is the inner product in the Hilbert space.  
For an RKHS, the inner product is given by the kernel $$K$$, so,
\\[
    DE^\*(f) = {\langle \cdot, DE(f) \rangle}\_K
\\]
for some $$DE(f) \in H_K$$.
This means, we can write:
\\[
    DE^\*(f)(h) = {\langle h, DE(f) \rangle}\_K
\\]
where $$DE(f) \in H_K$$, which is what we had to show!

### Conclusion
We have seen what it means to do functional gradient descent. One last question that one may have is: why do this in the first place? Every function can be parametrized, and we can do 'ordinary' gradient descent in the space of parameters, instead?

The answer is: yes, you always can! In general, you can parametrize any function in a number of ways, each parametrization gives rise to different steps (and different functions at each step) in gradient descent.

The advantage is that some loss functions that are non-convex when parametrized can be convex in the function space: this means functional gradient descent can actually converge to global minima, when 'ordinary' gradient descent could possibly get stuck.