---
layout: post
title:  "Making Sense of Finite Fields"
date:   2020-01-27 18:00:00 +0530
categories: galois finite fields
---

This is going to be a series of questions (that I had) with answers (that I found out/looked up) regarding Galois fields.
Our last post discussed that every Galois Field has $$p^n$$ elements for some $$n \in \mathbb{N}$$.

**Q.** Consider the Galois Field $$GF(p)$$. This is isomorphic to (ie, can be thought of as) $$\mathbb{Z}_p$$. What about $$GF(p^n)?$$ Is that isomorphic to $$\mathbb{Z}_{p^n}?$$  
**A.**  No! This is because $$\mathbb{Z}_{p^n}$$ has zero divisors, but $$GF(p^n)$$ cannot, as it is a field.

**Q.** Okay, so what are the elements of $$GF(p^n)?$$ How would you describe them?  
**A.**  They can be thought of as polynomials, each of whose coefficients come from $$\mathbb{Z}_p$$.

**Q.**  What are these polynomials? Where do they come from?  
**A.**  Actually, there is no 'canonical' representation of the elements of $$GF(p^n)$$ as polynomials. This is because $$GF(p^n)$$ is isomorphic to $$GF(p)[x]/{\langle f(x) \rangle}$$ where $$f(x)$$ is an irreducible polynomial of degree $$n$$. Depending on what $$f$$ is, you get different representations of each element!

**Q.** Interesting, but what is $$f$$ here? Why does it have to be irreducible? Why does it have to be of degree $$n?$$  
**A.**  Well, we can't work with $$GF(p)[x]$$ directly, because it is not a field! For example, the polynomial $$x$$ has no multiplicative inverse. (Evaluate at $$0$$ to show impossibility.) 
Note that the elements of $$GF(p)[x]/{\langle f(x) \rangle}$$  are in fact, sets containing polynomials. Polynomials $$a(x)$$ and $$b(x)$$ are in the same set iff the difference $$a(x) - b(x)$$ is a multiple of $$f(x)$$. As $$f$$ has degree $$n$$, we can characterize each set by a representative element $$c_{n - 1}x^{n - 1} + c_{n - 2}x^{n - 2} + \cdots + c_0$$. There are $$p$$ choices for each coefficient $$c_i$$ (as these come from $$\mathbb{Z}_p$$) and $$n$$ of them totally, so we have $$p^n$$ elements, as required. Each choice of $$c_i$$ gives rise to a distinct set. 
We also need no zero divisors in this construction. This follows exactly because $$f$$ is irreducible!

The choice of $$f$$ actually gives us a basis to represent $$GF(p^n)$$ - look at the coefficients $$c_i$$ above. As we know, changing the basis only changes the way we view elements, not the actual elements themselves.