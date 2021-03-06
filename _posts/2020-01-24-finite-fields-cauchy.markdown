---
layout: post
title:  "Finite Fields and Cauchy's Theorem"
date:   2020-01-24 18:00:00 +0530
categories: galois finite fields cauchy
---

Today, we were studying Galois Fields, and I've realized I've forgotten almost everything from Modern Algebra that I took almost a year ago.

I've always known that fields are quite restricted because of the many constraints on the operations on the field elements. (Compare to a group which can be really anything).  But I've never studied fields in depth. Today I learned that a finite field (a Galois field) has order $$p^n$$ for some prime $$p$$. This is extremely restrictive!

One of the easiest ways to prove this is by Cauchy's Theorem. (I had forgotten all about this.) 
Cauchy's Theorem states that:
> If a prime $$p$$ divides the order (size) of a group $$G$$, then $$G$$ must have an element of order $$p$$.

Recall the order of an element is the smallest number of times we must repeatedly apply the group operation on it to get the identity of the group. For a finite group, every element has finite order, because the group is closed under the group operation. In fact, Lagrange's theorem states that the order of an element always divides the order of the group!

Anyways, back to Cauchy's Theorem. I will only provide here the proof for when $$G$$ is an abelian group (when the group operation is commutative). Why? 
* The proof is simpler. The proof that I know of the non-abelian case requires knowledge of the Class Equation for Groups.
* We won't need the more general case for the proof of the above theorem on finite fields.

But you can easily find a full proof of Cauchy's Theorem [online](https://www.google.com/url?sa=t&source=web&rct=j&url=https://kconrad.math.uconn.edu/blurbs/grouptheory/cauchypf.pdf&ved=2ahUKEwiZxcmKypznAhXYF3IKHVfvCVEQFjATegQIAxAB&usg=AOvVaw1SD38-sQRiiHy_7i4GSMpZ&cshid=1579880752095).

#### Proof of Cauchy's Theorem for Abelian Groups
We use induction on the size of the group $$G$$. The case when $$\lvert G \rvert  = p$$ is easy. 
An abelian group $$G$$ has the property that every subgroup is normal.  Take a non-identity element $$g \in G$$. If $$p$$ divides the order of $$g$$, we're done, because if $$\lvert x\rvert = mp, \lvert x^m\rvert = \frac{mp}{\gcd(mp, m)} = p$$, making $$x^m$$ the element we want.

Otherwise, consider the subgroup $$H = \langle g \rangle$$ generated by it. We know this is a normal subgroup, so we can quotient by this to get a group $$G/H$$. $$p$$ divides $$\lvert G/H \rvert$$ (as this is equal to $$\lvert G\rvert / \lvert H\rvert$$ by Lagrange's Theorem), and the group $$G/H$$ is definitely smaller than $$G$$ (as $$g$$ is not identity), so the induction hypothesis says that there must be an element of order $$p$$ in $$G/H$$.

Now, every element in $$G/H$$ is of the form $$aH$$ for some $$a \in G$$. If $$aH$$ has order $$p$$ in $$G/H$$, we can write:
\\[
(aH)^p = a^p H = H = eH.
\\]
We used $$aH \times bH$$ = $$abH$$ repeatedly. Now, $$aH = bH \implies ab^{-1} \in H$$, so this means $$a^p e^{-1} = a^p \in H$$. Note that, in general, $$\lvert a^p\rvert = \frac{\lvert a\rvert}{\gcd(\lvert a\rvert, p)} \leq \lvert a\rvert.$$ 

But we cannot have equality! Because, $$\langle a^p \rangle \leq H$$ but $$\langle a \rangle \not\leq H$$ as $$a \not\in H$$ because $$aH \neq H$$. 

Then, as $$\lvert a^p\rvert < \lvert a\rvert$$, so $$\gcd(\lvert a\rvert, p) > 1 \implies p$$ divides $$\lvert a\rvert.$$ We can then apply the induction hypothesis on $$\langle a \rangle$$.  
This concludes the proof.


The theorem on finite fields follows now from the following facts:
* A finite field $$(F, +, \times)$$ has prime characteristic $$p$$. This follows from the fact that a field has no zero divisors.
* Every element in the additive group $$(F, +)$$ has order $$p$$. This follows from the definition of a characteristic and using distributivity of multiplication over addition.

Why? Note that as $$(F, +, \times)$$ is a field, $$(F, +)$$ is an abelian group by the field axioms. If a prime $$q \neq p$$ divided the order of $$(F, +)$$, by Cauchy's Theorem, there must be an element in $$(F, +)$$ that had order $$q$$, a contradiction to the second fact above.

Thus, $$\lvert F\rvert = \lvert (F, +)\rvert = p^n$$ for some $$n$$!

I'm also starting to introduce a commenting facility via GitHub Issues, in order to not clutter up this space here. Comment [here](https://github.com/simple-complexities/simple-complexities.github.io/issues/7)!