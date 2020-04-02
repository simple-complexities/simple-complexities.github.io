---
layout: post
title:  "Max-Flow Min-Cut with Linear Programming"
date:   2020-01-23 18:00:00 +0530
categories: linear program maxflow mincut
---
This work is based on [Totally Unimodular Matrices](https://theory.stanford.edu/~jvondrak/MATH233B-2017/lec3.pdf). However, they use a different notation there, so I just wanted to rewrite their proof in my own words.

Define the incidence matrix $$A$$ of a directed graph $$G$$ as:
\\[
    A_{ij} =  \begin\{cases\}
        1, & \text{if edge } j  \text{ originates at vertex } i \newline
        -1, & \text{if edge } j \text{ ends at vertex } i \newline
        0, & \text{otherwise} \newline
        \end\{cases\}
 \\]
Thus, the dimensions of $$A$$ are $$\lvert V \rvert \times \lvert E \rvert $$.

Now, let's move on to flows. Let $$c$$ be the capacity vector with all non-negative integer entries.
Let the row in $$A$$ corresponding to the source vertex $$s$$ be $$w$$.  Let $$A'$$ be $$A$$ with rows corresponding to $$s$$ and $$t$$ removed.
Then, maximum flow can be written as the primal linear program:

\\[ 
\begin\{aligned\}
       \max w^Tf \newline
       \text{such that }  f &\leq c, \newline
        f &\geq 0, \newline
       A'f &= 0.
\end\{aligned\}
\\]

Then, the dual linear program corresponds to:
\\[
\begin\{aligned\}
       \min c^Td \newline
       \text{such that } d &\geq 0, \newline
        z &\in \mathbb{R}, \newline
        A'^Tz + d &\geq w.
\end\{aligned\}
\\]

$$z$$ is actually a vector of size $$\lvert V \rvert  - 2$$. It has one variable for each vertex that is not $$s$$ nor $$t$$. This is inconvenient, so we add $$z_s = -1$$ and $$z_t = 0$$.

This changes the first dual constraint to $$A^Tz + d \geq 0$$. 
Now, we need some facts (that will not be proven here but these proofs are available in the link above):
* A matrix $$M$$ is said to be totally unimodular if every square submatrix of $$M$$ has determinant $$-1, 0$$ or $$1$$.
* A non-singular submatrix of a totally unimodular matrix $$M$$ has an integer inverse.
* The vertices of the convex polytope $$Mx \leq b$$ where $$b$$ has integer entries have all integer entries too.
* $$A$$ is totally unimodular for any graph $$G$$.

Note that both the primal and dual are feasible, hence, both have optimal solutions. The strong law of LP duality tells us that the optimal values are actually the same. 

Note that for an LP, a vertex of the feasible region must be in the optimal solution set (think of the gradient of the objective function and how it 'pulls').This shows that an optimal flow exists with integer values if all capacities are integers!

Thus, we can consider the optimal solutions of the primal and dual to be occurring at vertices of the respective feasible regions. Let $$f^*$$ and $$(z^*, d^*)$$ be the integer-valued optimal solutions respectively.
We have, looking at the constraint in the dual for edge $$(u, v)$$, and the feasibility condition,
\\[ 
d^\*\_{uv} \geq \max(0, z^\*\_v - z^\*\_u).
\\]
As the dual is a minimization problem, and $$c \geq 0$$, it will always be optimal to choose equality in the above inequality. However, we will not need this fact directly.

We need to relate these variables in the dual to a $$s-t$$ cut separating $$s$$ and $$t$$.  
Define $$S = \{ u \in V, z^*_u \leq -1\}$$. Then, $$s \in S$$ and $$t \in V - S$$, so this is a valid $$s-t$$ cut.

Consider an edge $$(u, v)$$ crossing this cut. As $$z^*$$ are all integers, the above gives (with the definition of $$S$$ and $$V - S$$):
\\[
d^\*\_{uv} \geq 1.
\\] 
Thus, the optimal value of the dual is 
\\[
c^Td^\* \geq \sum_{e \text{ crossing } } c_e = \text{ capacity of } s-t \text{ cut }\geq \text{ maximum flow out of } S = w^Tx^\*,
\\]
which is the optimal value of the primal.

But, we know that these optimal values are the same! Hence, we must have equality everywhere. This means:
\\[
    d\_{ij} =  \begin{cases}
        1, & \text{if  }  i \in S \text{ and } j \in T \newline
        0, & \text{otherwise} \newline
        \end{cases}
\\]
\\[
 z\_{i} =  \begin{cases}
        -1, & \text{if  }  i \in S \newline
        0, & \text{otherwise} \newline
        \end{cases}
\\]

I'm also starting to introduce a commenting facility via GitHub Issues, in order to not clutter up this space here. Comment [here](https://github.com/simple-complexities/simple-complexities.github.io/issues/9)!