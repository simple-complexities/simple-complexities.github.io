---
layout: post
title:  "A Review of Constrained Optimization Theory"
date:   2020-03-03 18:00:00 +0530
categories: optimization constrained theory
---

**In progress! Check back again soon!**    
I took a really good class on Optimization last year, but I seem to have forgotten a lot of the details of the theory we learned. This post is an effort to get back in touch with the material.

In this post, I closely follow the textbook ['An Introduction to Optimization'](https://www.amazon.com/Introduction-Optimization-Edwin-K-Chong/dp/1118279018) by Edwin Chong and Stanislaw Zak.

We're dealing with non-linear constrained optimization problems, with both equality and inequality constraints.
\\[
    \begin\{aligned\}
    \min f(x) \newline
    \text{such that } h(x) &= 0, \newline
    \text{and } g(x) &\leq 0.
    \end\{aligned\}
\\] 
The feasible set $$Feas$$ is the set of all feasible points:
\\[
    Feas = \\{x \mid h(x) = 0, g(x) \leq 0 \\}
\\]

Note that maximizing $$f$$ is the same as minimizing $$-f$$, so we do not lose anything by considering only minimization problems.

The form above is the most general form of the problem. Often, simplifying the problem helps in understanding. To this end, we will initially consider only equality constraints. But first, we will review the definition of a derivative.

**Important**: We're assuming all second-order derivatives exist and are continuous. This helps make the exposition clearer, and we don't have to worry about 'bad' functions.

### Review: The Jacobian of a Function
The derivative $$Df$$ of a multi-variable function $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ is defined by the transpose of its gradient vector:
\\[
    Df(x_1, x_2, \ldots, x_n) = \nabla f(x)^T = \left\[\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_i}, \ldots, \frac{\partial f}{\partial x_n}\right\]
\\]

What does the derivative mean for a multi-variable function? The idea is very similar to that for single-variable functions: the best linear approximation of $$f$$ about a point $$x$$: 
\\[
    \lim_{\\lVert h \rVert \rightarrow 0} \frac{\lVert{f(x + h) - (f(x) + Df(x)h)}\rVert}{\lVert{h}\rVert} = 0
\\]
Why we take the transpose of the gradient should be clear: we need $$Df(x)$$ to be a row vector!

The derivative $$Df$$ of a multi-variable function $$f: \mathbb{R}^n \rightarrow \mathbb{R}^m$$ is defined by the concatenation of the derivatives of each of its components.
\\[
    f(x) = \begin{bmatrix}
                f_1(x) \newline 
                \ldots \newline 
                f_m(x) \newline 
           \end{bmatrix}    
    \Rightarrow
    Df(x) = \begin{bmatrix}
                Df_1(x) \newline 
                \ldots \newline 
                Df_m(x) \newline 
            \end{bmatrix}
         = \begin{bmatrix}
                \nabla f_1(x)^T \newline 
                \ldots \newline 
                \nabla f_m(x)^T \newline 
            \end{bmatrix}
\\]

You can see that in this case, $$Df$$ is now an $$m \times n$$ matrix, called the Jacobian matrix of $$f$$ at $$x$$. 

---
**Example**:  
The Jacobian matrix $$Df$$ of
\\[
    f(x, y, z) = (x^2 + y^2, x^2 + y - z^2)
\\]
is given by:
\\[
    Df(x, y, z) = \begin{bmatrix}
                    Df_1(x, y, z) \newline 
                    Df_2(x, y, z) \newline 
                  \end{bmatrix}
                = \begin{bmatrix}
                    2x & 2y & 0 \newline
                    2x & 1   & - 2z \newline
                  \end{bmatrix}
\\]
where,
\\[
    \begin{aligned}
    f_1(x, y, z) &= x^2 + y^2 \newline
    f_2(x, y, z) &= x^2 + y - z^2 \newline
    \end{aligned}
\\]

---

You may also be wondering: what if I wanted to move beyond $$\mathbb{R}^n$$ and $$\mathbb{R}^m$$, and instead compute derivatives of functions on smooth surfaces? The answer is: you can! The term you're searching for is called the 'pushforward'. We won't be doing this here, but I wanted to share some insight about why this is called the 'pushforward', in the first place.

Think of what the derivative gives us, apart from the best linear approximation to a function.

The Jacobian matrix tells us that: if I start at $$x$$ and move along $$v$$ (a vector), my function would change by $$Df(x)v = J(v)$$ (a vector) approximately. Thus, it maps movement vectors (such as $$v$$) in one space to movement vectors (such as $$J(v)$$) in another space. In some sense, the Jacobian is showing us how to 'push' vectors from one space to the other.

Of course, generally on surfaces, we may not be able to move in every direction! Thus the pushforward is defined for vectors in the tangent space of the surface at the point, mapping these to vectors which are in the tangent space of the 'image' surface at the image of the point.

What is a tangent space, again?

### Review: Tangent and Normal Spaces
With the definition of a derivative now clear, we can begin to define the tangent space at a point $$x$$ on a surface $$S$$ defined by $$h(x) = 0$$:
\\[
    T(x) = \\{ y \mid Dh(x)y = 0 \\}
\\]
Why this definition? We stay close to $$S$$ for sufficiently small perturbations moving along any $$y \in T(x)$$.
We will need the particularly important fact:
For any curve on $$S$$, the tangent vector to the curve at $$x \in S$$ lies in the tangent space of $$S$$ at $$x$$.


The normal space at $$x$$ can be defined as the subspace spanned by the gradients:
\\[
    N(x) = \\{y \mid y = Dh(x)^Tv = \sum_i v_i \nabla h_i(x) \\}
\\]
We can check the orthogonality condition. If $$t \in T(x)$$ and $$n \in N(x)$$,
\\[
    n^Tt = v^TDh(x)t = v^T(\vec{0}) = 0. 
\\]

### Review: The Hessian of a Function
The Hessian is the multi-variable equivalent of the second derivative, just like how the Jacobian is the multi-variable equivalent of the first derivative.
However, since we have a choice of indices for the second derivatives, the Hessian of $$f: \mathbb{R}^n \rightarrow \mathbb{R}$$ is now a matrix:
\\[
    Hf(x) =  \begin{bmatrix}
                \frac{\partial^2 f}{\partial x_1 \partial x_1} & \ldots & \frac{\partial^2 f}{\partial x_1 \partial x_n} \newline
                \ldots & \frac{\partial^2 f}{\partial x_i \partial x_j} & \ldots \newline
                \frac{\partial^2 f}{\partial x_n \partial x_1} & \ldots & \frac{\partial^2 f}{\partial x_n \partial x_n} \newline
             \end{bmatrix}
\\]
and the Hessian of $$f: \mathbb{R}^n \rightarrow \mathbb{R}^m$$ is now a 'third-order' tensor (but we will not use this here, luckily!)


### Review: Conditions for Unconstrained Optimization
Unconstrained optimization is a lot easier to handle (and is also more familiar to everyone).
I'm not going to be proving the conditions here, but they aren't too hard to show. They are the equivalent of the first and second order tests for single-variable functions!

#### First-Order Necessary Conditions (FONC)
If a point $$x^*$$ is a local minimizer of $$f$$, then:
\\[
    Df(x^\*) = 0^T.
\\]
Transposing, this is the same as saying:
\\[
    \nabla f(x^\*) = 0.
\\]

#### Second-Order Necessary Conditions (SONC)
If a point $$x^*$$ is a local minimizer of $$f$$, then:
\\[
    \begin{aligned}
        Df(x^\*) &= 0^T, \newline
        Hf(x^\*) &\text{ is positive semi-definite.} \newline
    \end{aligned}
\\]

#### Second-Order Sufficient Conditions (SOSC)
If a point $$x^*$$ is such that:
\\[
    \begin{aligned}
        Df(x^\*) &= 0^T, \newline
        Hf(x^\*) &\text{ is positive definite.} \newline
    \end{aligned}
\\]
then $$x^*$$ is a local minimizer of $$f$$.

As you can see, these conditions for unconstrained optimization are really simple. Can we get equivalents for constrained optimization too?

### Problems with Equality Constraints Only

We can finally start!
A regular point is defined as a feasible point where all the gradient vectors of the constraint functions $$h_i$$ are linearly independent. This is equivalent to saying that $$Dh(x)$$ is of full rank, if $$x$$ is regular.

Regularity plays an important role!

### Lagrange's Theorem (and Lagrange Multipliers)

If $$x^*$$ is a local minimizer of $$f$$, $$x^*$$ being a regular point, then there exists $$\lambda^*$$ such that:
\\[
    Df(x^\*) + {\lambda^*}^T Dh(x^\*) = 0^T.   
\\]

Taking transposes, this is saying that $${Df(x^*)}^T$$ is in the normal space! (With respect to $$Feas$$, of course.)
\\[
    {Df(x^\*)}^T + {Dh(x^\*)}^T{\lambda^\*} = 0.
\\]

The $$\lambda^*$$ are called the Lagrange Mutlipliers.

How does one prove this? Consider any curve parametrized by $$t$$ through $$x^*$$. Look at the one-dimensional function $$\phi(t) = f(x(t))$$. $$\phi$$ has a local minima at $$t = t^*$$ (when we are at $$x = x^*$$), so the derivative of $$\phi$$ at $$t = t^*$$ must be $$0$$. Applying the chain rule, and using the first-order necessary conditions for unconstrained optimization ($$\phi$$ is not constrained!):
\\[
    {\frac{d\phi}{dt}}(t^\*) = Df(x(t^\*))\dot x(t^\*) = 0
\\] 
But $$\dot x(t^*)$$ is a tangent vector of the curve, so it is in the tangent space. As $$Df(x^*)^T$$ is perpendicular to this, it must be in the normal space (span of the gradient vectors). The coefficients of the gradient vectors in the linear combination is given by $$\lambda^*$$.

This gives us the equivalent of FONC for constrained optimization problems!