---
title: "From Mixture of Experts to Mixture of Agents: Sparse Routing Is Escaping the Model"
excerpt: "The idea that powers today's frontier models is a routing idea — and routing was never going to stay inside the model."

header:
  image: "./../assets/images/posts/2026-06-22-from-mixture-of-experts-to-mixture-of-agents/moe-hero.svg"
  teaser: "./../assets/images/posts/2026-06-22-from-mixture-of-experts-to-mixture-of-agents/moe-hero.svg"
  caption: "One gate, many experts — and the same principle, one level up"

tags: [mixture-of-experts, agent-matrix, inference, architecture]
essay: true
essay_date: "2026-06-03"
audio_slug: mixture-of-agents
audio: true
read_time: "11 min read"
card_image: /assets/images/posts/2026-06-22-from-mixture-of-experts-to-mixture-of-agents/moe-hero.svg
card_excerpt: "The idea that powers today's frontier models is a routing idea — and routing was never going to stay inside the model."
---

<!-- audio:start -->

Almost every frontier model you have used in the last two years owes its economics to a single architectural trick, and almost no one is writing about where that trick is heading next. The trick is the Mixture of Experts, and the reason it matters is not really that it makes models bigger. It is that it makes them *sparse* — and sparsity, once you understand what it is for, turns out to be a principle far too useful to leave trapped inside a neural network.

Let me explain the trick cleanly first, because the forward-looking argument only lands if the foundation is exact.

## What a Mixture of Experts actually does

In a conventional transformer, every token that flows through the network passes through the same dense feed-forward layer. Every parameter in that layer participates in every token. This is simple and it is wasteful: you are paying, on every single token, for the entire capacity of the model, whether the token needs it or not.

A Mixture of Experts replaces that one dense layer with many smaller "expert" networks sitting side by side — eight of them, or sixty-four, or more — and adds a small **gating network** whose only job is to decide which experts should handle each token. The gate scores the experts for the token in front of it and routes the token to the best few: this is **top-k routing**, and in practice *k* is small, usually one or two. A token activates its top-k experts and ignores the rest entirely.

<figure class="post-figure">
  <img src="/assets/images/posts/2026-06-22-from-mixture-of-experts-to-mixture-of-agents/dense-vs-moe.svg" alt="Dense feed-forward versus Mixture of Experts. On the left, a single solid FFN block: every token activates every parameter. On the right, a small gate routes each token to two of six experts; the rest stay dark." loading="lazy">
  <figcaption>Left: a dense layer charges full price for every token. Right: a gate picks two experts out of six — total capacity scales with N, cost per token stays pinned to k.</figcaption>
</figure>

The consequence is the part worth internalising, because it is genuinely counter-intuitive. The model's *total* capacity now scales with the number of experts, but the *compute per token* is fixed by k, not by the total. A so-called 8×7B model with top-2 routing carries on the order of fifty-six billion parameters in total, yet activates only around fourteen billion for any given token. You get the knowledge capacity of a very large model at roughly the inference cost of a small one. Same FLOPs as a 14B dense network; far more room to store what the model knows. Total parameters and active parameters have been decoupled, and that decoupling is the whole game.

> Sparsity is not a way to make a model bigger. It is a way to stop paying for everything on every token.

The decoupling is easier to feel than to read. The simulation below shows it directly: **N** is the total bench of specialists, **k** is how many fire on each request, and *capacity / compute* is the ratio you stop paying for. Drag the sliders — push N up and watch capacity climb while the cost per request stays pinned to k. Then drop **Balance** below 45% and the system **collapses**: most traffic piles onto the same few specialists and the rest go dark. This is the same dial that, scaled up, decides whether a multi-agent system stays healthy or quietly becomes a monolith.

<figure class="post-figure post-figure--sim">
  <iframe src="/assets/sims/moe-routing.html" title="Interactive sparse-routing simulation: total specialists N, active per step k, load balance, and request speed" loading="lazy" allowfullscreen></iframe>
  <figcaption>Try it: increase <strong>N</strong> to grow capacity, hold <strong>k</strong> low to keep compute flat, drop <strong>Balance</strong> below 45% to see collapse, click <em>Send one request</em> to step through manually.</figcaption>
</figure>

It is not free, of course. Two failure modes haunt every MoE system, and both are really failures of routing. The first is **expert collapse**: the gate discovers a couple of experts that work well enough, sends them everything, and lets the others wither — capacity you paid for but never use. The second is its mirror image, **dead experts**: specialists that the gate never learns to route to, so they sit idle forever. The standard remedies are equally instructive. You add an **auxiliary load-balancing loss** that penalises the gate for piling traffic onto a few favourites, pushing it to spread work across the bench. And you add a little noise to the gate's scores — **noisy top-k** — so that ties break fairly and every expert gets enough of a chance to prove useful. Keep those two forces in tension and the system stays healthy: every expert earns its keep, and no single one hogs the line.

<figure class="post-figure">
  <img src="/assets/images/posts/2026-06-22-from-mixture-of-experts-to-mixture-of-agents/collapse-vs-balanced.svg" alt="Two bar charts. Left: traffic skewed onto two experts while the other six are flatlined — collapse. Right: traffic spread evenly across all eight experts — balanced." loading="lazy">
  <figcaption>Collapse (left) is what you get without active load-balancing: a couple of favourites carry the system. Balanced routing (right) keeps every specialist sharp — and it is the discipline you re-derive at every scale of this pattern.</figcaption>
</figure>

Hold onto those two failure modes. We are going to meet them again, somewhere you might not expect.

## Routing was never a property of models

Here is the turn, and it is the reason I wanted to write this at all. Everything I just described is presented, in paper after paper, as a *model* technique — a clever thing you do between the attention layers. But look at what the mechanism actually is, stripped of its setting. You have a population of specialists. You have a gate that, for each incoming unit of work, decides which specialists should handle it. You activate a few and leave the rest dormant, so total capacity grows while cost per unit stays flat. That is not a fact about transformers. That is a general principle for organising any system where you have more capability than you can afford to use all at once.

So ask the obvious question: why should the experts be sub-networks? Why should the gate live inside a single model's forward pass? Nothing in the principle requires it. Lift the whole pattern up one level of abstraction and the "experts" become entire models — or entire agents — and the "gate" becomes a system that decides which one should handle a given task. A small local model for the easy, high-volume work. A frontier API for the genuinely hard cases. A fine-tuned domain specialist for the narrow, high-stakes niche. Most requests never touch the expensive option, exactly as most tokens never touch most experts. The economics are the same because the principle is the same.

And this is not hypothetical. It is already running. When a unifying gateway sits in front of many language-model backends and decides, per request, whether to serve it from a local Ollama instance, a free cloud, or a paid API, that gateway *is* a gate performing top-k routing over a mixture of experts — the experts are just whole models now. That is precisely what [OllaBridge](https://ollabridge.com) does: one OpenAI-compatible endpoint in front of every model you can run, routing each call to the backend that should answer it. The sparse-routing pattern climbed out of the transformer and became a piece of infrastructure.

Push the abstraction one more level and the experts stop being models at all. They become *agents* — specialised autonomous workers, each good at something, coordinated by a system that routes tasks to whichever agent is right for the job. A multi-agent operating system that dispatches work to the correct specialist agent is running the identical pattern at the highest level yet: a gate, a population of experts, sparse activation, capacity that scales with the roster while cost stays bounded. This is the routing layer at the heart of [Agent-Matrix](https://agent-matrix.github.io), and once you have seen the lineage — token to model to agent — it is hard to unsee. It is one idea wearing three different sizes.

<figure class="post-figure">
  <img src="/assets/images/posts/2026-06-22-from-mixture-of-experts-to-mixture-of-agents/three-scales.svg" alt="Three side-by-side panels showing the same router-and-experts shape at three scales: tokens routed to sub-networks, requests routed to whole models, tasks routed to autonomous agents." loading="lazy">
  <figcaption>The same shape at three altitudes. Only the size of the "expert" changes — from sub-network, to whole model, to autonomous agent — and with it the unit of work being routed.</figcaption>
</figure>

## The same failures, scaled up

The clearest sign that this is genuinely the same principle, and not a loose metaphor, is that the failure modes come along for the ride. Expert collapse, at system scale, is the situation where one model or one agent ends up handling everything because the router found it "good enough," while the specialists you built and paid for sit idle — you have rebuilt a monolith out of parts that were supposed to be a system. Dead experts, at system scale, are the capabilities that never get used: the domain agent nobody routes to, the cheap local model the gateway always bypasses, the specialist quietly rotting on the bench. Same pathologies, larger blast radius.

And the fixes rhyme too. You balance the load deliberately rather than letting the router converge on a lazy favourite. You route on purpose — with policy, with cost-awareness, with a bias toward giving every capable specialist enough traffic to stay sharp — which is the system-level echo of the load-balancing loss and noisy top-k that keep an MoE layer healthy. The discipline you learn building a good Mixture of Experts is, almost line for line, the discipline you need to build a good mixture of agents.

I want to be precise about the claim, because the analogy is powerful enough to be abused. A gateway routing across model backends is not literally a Mixture of Experts model; there is no shared gradient, no joint training, no differentiable gate learning the routing end to end. What I am claiming is narrower and, I think, more interesting: it is the *same principle at a different scale*. Sparse, learned-or-engineered routing over a population of specialists is a pattern that is valid from the feed-forward layer all the way up to a fleet of autonomous agents, and recognising that lets you carry hard-won lessons across the levels instead of relearning them each time.

## Where this goes

The instinct of the last few years has been to chase one enormous model that can do everything. Mixture of Experts was the first crack in that instinct — the admission that even inside a single model, it is better to have many specialists and route between them than to make every parameter do every job. I think that admission is a preview, not an exception. The efficient AI systems of the next several years will not be one giant model. They will be well-routed systems of many: small models and large ones, general agents and narrow ones, cheap paths and expensive paths, with a gate deciding, request by request, where each piece of work should go.

The hard part of building those systems is not the models. The models are abundant and getting cheaper. The hard part is the routing — the gate, the balancing, the deliberate avoidance of collapse and decay — and that is a discipline the field has already been quietly practising, at small scale, inside every MoE layer it has trained. Sparse routing is escaping the model. The teams that understand it as an architecture for systems, and not just a trick for layers, are the ones who will build what comes next.

<!-- audio:end -->

---

*The model-level and agent-level routing in this essay are not theoretical: [OllaBridge](https://ollabridge.com) routes across backends today, and [Agent-Matrix](https://agent-matrix.github.io) routes across agents. If you want the deeper architecture, the [Agent-Matrix essay](/blog/Matrix-the-first-alive-AI-system) lays out the system this routing layer lives inside.*
