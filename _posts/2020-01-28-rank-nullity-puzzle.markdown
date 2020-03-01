---
layout: post
title:  "Solving a Logical Puzzle with Rank-Nullity"
date:   2020-01-28 18:00:00 +0530
categories: rank nullity puzzle
---

This post is in a different flavour, altogether. This is an interesting problem someone posted online:

> You have a set of $$2n+1$$ stones with positive real masses. Suppose that every subset of size $$2n$$ of these stones can be split into $$2$$ sets of equal mass, each containing $$n$$ stones. Prove that all stones have the same mass.


The solution is quite nice, hence the post. 
Consider the matrix $$A$$ where the $$i^\text{th}$$ row is formed by removing the $$i^\text{th}$$ stone, and indicating $$a_{ij} = \pm 1$$ which group the $$j^\text{th}$$ stone belongs to ($$a_{ii} = 0$$ as this stone does not take part).

Note that the condition that each subset has $$n$$ stones gives us that each row of $$A$$ sums up to $$0$$.
Therefore, the all-ones vector $$\vec{1}$$ of size $$2n + 1$$ is in the null-space of $$A$$.
\\[ A \times \vec{1} = \vec{0} \\]
As the dimension of its null-space is atleast $$1$$, rank-nullity tells us that $$A$$ can have rank atmost $$2n$$. We will now show that $$A$$ has rank exactly $$2n$$. Why does this prove the required result? Because the weight vector $$\vec{w}$$ by definition must be in the null-space of $$A$$. But the null-space of $$A$$ is spanned by $$\vec{1}$$! This means all the weights are equal as $$\vec{w} = w \cdot \vec{1} $$!

Consider the submatrix $$B$$ obtained by deleting the last column and last row of $$A$$. Thus, $$B$$ has size $$2n \times 2n$$. The crucial property of $$B$$ that we will use is that $$B$$ is invertible! 

#### Proof that $$B$$ is invertible
It is sufficient to show that $$B$$ has determinant not equal to $$0$$. This can be done by considering the determinant of the matrix $$B'$$ where $$B'$$ contains the modulo $$2$$ values of the respective entries in $$B$$. Thus, every entry in $$B'$$ is $$0$$ or $$1$$. Note that the determinant of $$B$$ and $$B'$$ are the same modulo $$2$$. 

Now, $$B' = J - I$$, where $$J$$ is the all-ones matrix, and $$I$$ is the identity matrix. $$J$$ has eigenvalue $$0$$ of multiplicity $$2n - 1$$ (as its rank is $$1$$) and eigenvalue $$2n$$ of multiplicity $$1$$. Thus, $$B' = J - I$$ has eigenvalue $$-1$$ of multiplicity $$2n - 1$$ (as its rank is $$1$$) and eigenvalue $$2n - 1$$ of multiplicity $$1$$. 

Why do we subtract $$1$$ from the eigenvalues when we subtract $$I$$ from the matrix $$J$$? Look at the defining equation of an eigenvalue-eigenvector pair!

\\[
    Ax = \lambda x \implies (A - I)x = (\lambda - 1)x
\\]

The determinant of a matrix is the product of its eigenvalues, so $$\det(B') = (2n - 1)(-1)^{2n - 1} $$ $$ = -(2n - 1)$$. But, then modulo 2, $$\det(B) \equiv \det(B') \equiv 1$$. This shows that $$\det(B)$$ is not $$0$$, hence $$B$$ is invertible. 


Then, $$B$$ has rank $$2n$$ as it is full-rank. On adding the deleted row and deleted column back to get $$A$$, the rank is unaffected. (In general, the rank of a matrix is $$\geq$$ the rank of any submatrix). Hence, the rank of $$A$$ is $$2n$$, and following above, all the weights are equal.