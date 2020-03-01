---
layout: post
title:  "Interlacing of Eigenvalues"
date:   2020-02-10 18:00:00 +0530
categories: eigenvalue interlacing
---

Cauchy's Interlacing Theorem says that the eigenvalues of submatrix are interlaced between the eigenvalues of the original real symmetric matrix.
This is a pretty famous result.

Another interesting interlacing result that we did in class a few weeks was:

>Let $$B$$ be a real symmetric matrix. Define:
\\[
A = B + xx^T
\\]
If the $$n$$ eigenvalues of $$A$$ are $$\lambda_1 \geq \ldots \geq \lambda_n$$, and the $$n$$ eigenvalues of $$B$$ are $$\mu_1 \geq \ldots \geq \mu_n$$, then these are interlaced as:
\\[
\lambda_1 \geq \mu_1 \geq \ldots  \lambda_i \geq \mu_i \geq \lambda_{i + 1} \ldots \lambda_n \geq \mu_n.
\\]

The proof (which we rushed through in class, and hence this post) consists of two distinct parts:
* $$\lambda_i \geq \mu_i$$.  
For any $$y \in \mathbb{R}^n$$, we have:
\\[ y^TAy = y^TBy + y^Txx^Ty = y^TBy + (x^Ty)^2 \geq y^TBy. \\]
Now, note the Rayleigh quotient characterization of eigenvalues:
\\[
\lambda_i = \max_{\substack{U \leq \mathbb{R}^n \\ \dim U = i}} \min_{\substack{y \in U \\ |y| = 1}} (y^TAy)
\\]\\[
\mu_i = \max_{\substack{U \leq \mathbb{R}^n \\ \dim U =  i}} \min_{\substack{y \in U \\ |y| = 1}} (y^TBy)
\\]
There is also a minimax formulation:
\\[
\lambda_i = \min_{\substack{U \leq \mathbb{R}^n \\ \dim U = n - i + 1}} \max_{\substack{y \in U \\ |y| = 1}} (y^TAy)
\\]\\[
\mu_i = \min_{\substack{U \leq \mathbb{R}^n \\ \dim U = n - i + 1}} \max_{\substack{y \in U \\ |y| = 1}} (y^TBy)
\\]
We can use either, but we will only use the first here.
We are now done by noting that if $$f \geq g$$ everywhere on a common domain $$X$$, then both $$\min f \geq \min g$$  and $$\max f \geq \max g$$, applying this twice.

* $$\mu_i \geq \lambda_{i + 1}$$.  
This is the harder part. The proof takes an idea similar to that for Cauchy's interlacing theorem though! Let us define the subspace $$V$$ of dimension $$n - 1$$: \\[ V = \\{ y \in \mathbb{R}^n \mid x^Ty = 0 \\} \\]
Consider a subspace $$U$$ of $$\mathbb{R}^n$$ with dimension $$i$$. From this, create the subspace $$U' = U \cap V$$. 

Note that $$\dim(U')$$ can be only $$i - 1$$ or $$i$$. How? Use the general fact that:
\\[ n \geq \dim(U + V) = \dim(U) + \dim(V) - \dim(U \cap V) \\] 
and $$\dim(U') \leq \dim(U)$$.

Now, for every $$y \in U',$$
\\[ y^TAy = y^TBy + y^Txx^Ty = y^TBy + (x^Ty)^2 = y^TBy. \\]

Then,  as the minimum over a subset cannot be smaller than the minimum over a superset,
\\[ \min_{y \in U'} y^TBy = \min_{y \in U'} y^TAy  \geq \min_{y \in U} y^TAy. \\]
But we also have:
\\[ \mu_i  = \max_{\substack{S \leq \mathbb{R}^n \\ \dim S \geq  i}} \min_{\substack{y \in S \\ |y| = 1}} y^TBy \geq \min_{y \in U'} y^TBy \geq \min_{y \in U} y^TAy. \\]
for all $$U$$.
(Replacing $$\dim S = i$$ in the original formulation above by $$\dim S \geq  i$$ does not change anything!)

Thus,
\\[ \mu_i  \geq \max_{\substack{U \leq \mathbb{R}^n \\ \dim U = i + 1}}\min_{y \in U} y^TAy = \lambda_{i + 1}. \\]
and we're done!