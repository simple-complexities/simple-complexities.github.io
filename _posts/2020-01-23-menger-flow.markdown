---
layout: post
title:  "Menger's Theorem with Max-Flow Min-Cut"
date:   2020-01-23 18:00:00 +0530
categories: menger maxflow mincut
---

Once we have proved Max-Flow Min-Cut, we can use this to prove a variety of results. The work below can be thought of as a fleshing out of the ideas in [url=http://web.math.ucsb.edu/~padraic/notes_mathcamp.html]Flows in Graphs[/url].

Menger's Theorem states that the minimum number of edges whose removal is required to separate vertices $$s$$ and $$t$$ in an undirected graph $$G$$ is equal to the maximum number of edge-disjoint paths from $$s$$ to $$t$$.

A preliminary note is that, given a set of edge-disjoint paths from $$s$$ to $$t$$, we need to remove at least one edge from each of the paths in this set. Otherwise, we could just use the untouched path to go from $$s$$ to $$t$$. Menger's Theorem says that this is sufficient.

Let us create a flow network $$G_F$$ from $$G$$ by splitting undirected edges $$\{x, y\}$$ into directed edges $$(x, y)$$ and $$(y, x)$$, and assigning each directed edge, a capacity of $$1$$.

We prove this by two lemmas:
* $$G_F$$ has a $$0- 1$$ flow of value $$k$$ iff there are $$k$$ edge-disjoint paths from $$s$$ to $$t$$ in $$G$$.
Note that the 'if' direction is easy: the $$k$$ paths never intersect, so a flow of value $$k$$ exists.
For the 'only if' direction, we show something even stronger: we can find $$k$$ edge-disjoint paths using only the corresponding edges in $$G_F$$ where the flow is $$1$$.   
The idea is to use induction on the tuple $$($$size of set of edges where flow $$ = 1, k)$$. Starting from $$s$$, keep following an edge with flow $$1$$. When we visit intermediate nodes that are not $$s$$ or $$t$$, there must be an outgoing edge with flow $$1$$ because of flow conservation. Continuing, we either reach $$t$$, or cycle back to a vertex already seen. In either case, we can zero out the flow (along the $$s - t$$ path,  or the cycle), and use the induction hypothesis.
* If an $$s-t$$ cut in $$G_F$$ has capacity $$k$$, then removing $$k$$ edges is enough to disconnect $$s$$ and $$t$$.
Each edge crossing the cut from $$S$$ to $$T$$ has a capacity of $$1$$, so just removing these edges will make it impossible to go from $$s \in S$$ to $$t \in T$$.

We will need the contrapositive of the second lemma: if $$k$$ edges are required to be removed to disconnect $$s$$ and $$t$$, every cut must have capacity $$\geq k$$.