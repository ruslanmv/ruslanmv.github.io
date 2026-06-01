---
title: "All Projects"
permalink: /projects/all
layout: projects-all
sitemap: true
canonical_url: https://ruslanmv.com/projects/all
hero_title: "All Projects"
hero_image: /assets/images/header-projects.jpg
pj_intro: "A complete archive of research, engineering, AI, and scientific-computing projects."
header:
  og_image: /assets/images/header-projects.jpg
---

I design, build, and ship systems that reason, act, and govern themselves. My focus over the last few years has moved from analysing data toward building autonomous platforms and live AI products used by people around the world, and the projects below are organized by the role they play, from the open-source infrastructure that underpins everything, through the products built on top of it, to the experiments and the research foundations in physics and high-performance computing that taught me to make complex systems behave reliably.

---

*Platforms and infrastructure — the open-source core that everything else is built on.*

## Agent-Matrix

[![Agent-Matrix](/assets/images/posts/projects/agent-matrix.png)](https://agent-matrix.github.io "Agent-Matrix")

Agent-Matrix is an open-source operating system for governed, self-healing autonomous AI, and it is the project I care about most. Rather than treating an agent as a single script that runs and stops, it treats intelligence as something alive: a continuous loop in which the system perceives a problem, plans a response, checks that response against safety and policy, executes it, and learns from the result. I designed its core organs around that idea, including a registry and discovery hub, a governance and policy layer, a distributed execution engine, and a thermo-economic accounting system in which every unit of compute an agent consumes must be paid for in an energy-backed internal currency. That last piece comes directly from my physics background, where conservation laws are not optional, and it is, as far as I know, the only open implementation of its kind.

[Vision](https://agent-matrix.github.io) · [Source](https://github.com/agent-matrix) · [Live demo](https://cloud.matrixhub.io)

## MatrixHub

[![MatrixHub](/assets/images/posts/projects/matrixhub.png)](https://www.matrixhub.io "MatrixHub")

MatrixHub is the catalog and installer at the heart of Agent-Matrix, and the most production-ready part of the ecosystem. The simplest way to describe it is as a package manager for agents and tools, a place where an autonomous system can search for a capability it needs, install it with a reproducible and idempotent plan, and register it for use, all behind a clean API. Under the surface it combines hybrid search over both keywords and meaning, lockfile-style reproducibility, and native registration into an MCP gateway, so that agents can discover and adopt new tools the way a developer installs a library. It is backed by a live cloud instance you can experiment with directly, which makes the whole alive-system idea something you can touch rather than just read about.

[Product](https://www.matrixhub.io) · [Source](https://github.com/agent-matrix/matrix-hub) · [Live demo](https://cloud.matrixhub.io)

## OllaBridge

[![OllaBridge](/assets/images/posts/projects/ollabridge.png)](https://ollabridge.com "OllaBridge")

OllaBridge is an open-source AI gateway that unifies every language model you can run, whether a local Ollama instance, a spare GPU on a gaming machine, a free cloud notebook, or a paid commercial API, behind a single OpenAI-compatible endpoint. Its defining trick is that nodes dial outward over a WebSocket rather than waiting to be reached, so it works behind NAT, carrier-grade NAT, and corporate firewalls with nothing to port-forward and no VPN to configure. It pairs an installable local gateway, complete with a smart router, an admin dashboard, and a model catalog that scores and pulls the right models for your hardware, with a hosted control plane that handles device pairing, shared compute across machines, and the security expected of real deployments, including API keys, rate limits, OAuth, role-based access, and audit logs. It also ships an MCP server so agents can manage nodes and routes directly, which is how it quietly powers the avatars and personas elsewhere in my work. Released under Apache 2.0.

[Product](https://ollabridge.com) · [PyPI](https://pypi.org/project/ollabridge/) · [Source](https://github.com/ruslanmv)

---

*AI products — live, shipped applications people use today.*

## MedOS

[![MedOS](/assets/images/posts/projects/medos.png)](https://ai-medical-chabot.com "MedOS")

MedOS is a worldwide medical assistant that offers private, multilingual health guidance aligned with the published recommendations of bodies such as the WHO, the CDC, and the NHS. A person describes what is bothering them in their own language, and MedOS responds with clear, source-aligned information, available at any hour and designed to be approachable rather than clinical. It is built to inform and orient rather than to diagnose, and the emphasis throughout is on privacy and on grounding every answer in recognised public-health guidance, which is what makes it trustworthy as a first point of contact.

[Live product](https://ai-medical-chabot.com)

## LearnAI

[![LearnAI](/assets/images/posts/projects/learnai.png)](https://learnskillsai.com "LearnAI")

LearnAI is a personal AI tutor that adapts its lessons, practice, and feedback to each learner's age, pace, and goals, built on a single pedagogical loop that expresses itself through several distinct learning worlds. The same underlying engine teaches a six-year-old to count through playful stories, guides a teenager through exam preparation, and helps a working professional study cloud, coding, and AI with cited sources, while a calmer world serves older adults learning digital literacy at a gentler pace. It extends into an organizations tier with dashboards, learning paths, and reporting for schools and teams, which turns a personal tutor into something an institution can deploy. It is the project of mine with the broadest possible audience, since it is designed for every kind of learner rather than a single one.

[Live product](https://learnskillsai.com) · [For organizations](https://learnskillsai.com/organizations)

## GitPilot

[![GitPilot](/assets/images/posts/projects/gitpilot.png)](https://github.com/ruslanmv/gitpilot "GitPilot")

GitPilot is a multi-agent coding assistant that understands a whole repository rather than a single file. Built on a crew of specialised agents, it can read an existing codebase, reason about its structure, and generate or modify code with the surrounding context in mind, which is what separates it from a simple autocomplete. It is deliberately provider-agnostic, routing requests across several language-model backends so it is never locked to one vendor, and it is extensible through a plugin ecosystem so new capabilities can be added without rewriting the core. Of everything I have built, it is the developer tool with the broadest day-to-day usefulness.

[Source](https://github.com/ruslanmv/gitpilot)

---

*Interactive and experimental — where I try ideas that often become the seeds of a product.*

## HomePilot

[![HomePilot](/assets/images/posts/projects/homepilot.png)](https://github.com/ruslanmv/HomePilot "HomePilot")

HomePilot is a large modular application for building and running AI personas, assembled from FastAPI, React, and a combination of CrewAI and LangGraph for orchestration. It connects to real tools through a set of integration servers, so a persona can reason and then actually act in the world rather than only talk, and it includes a catalog of image-generation models alongside its conversational capabilities. Its persona engine drives a live, embodied interface, a browser-based 3D and VR avatar you can speak to with multi-provider language-model support and real-time speech, which you can try directly. It is the project where I explore what a genuinely useful personal AI system looks like when it is given real tools, a modular architecture, and a face.

[Live avatar](https://yourfriend.online) · [Source](https://github.com/ruslanmv/HomePilot) · [Avatar source](https://github.com/ruslanmv/3D-Avatar-Chatbot) · [How its personas &amp; memory work](/personas-and-memory/)

## VRSecretary

[![VRSecretary](/assets/images/posts/projects/vrsecretary.png)](https://github.com/ruslanmv/VRSecretary "VRSecretary")

VRSecretary is an embodied virtual assistant living inside Unreal Engine 5, a VR avatar you can speak to that answers with a real voice. A FastAPI backend connects the avatar to language models, while a text-to-speech layer gives it natural speech and the engine handles the embodiment and animation. Where the HomePilot avatar runs in the browser, VRSecretary pushes the same idea toward full game-engine fidelity, and it brings together my interests in avatars, real-time systems, and conversational AI in a single immersive prototype.

[Source](https://github.com/ruslanmv/VRSecretary)

## GamePilot

[![GamePilot](/assets/images/posts/projects/gamepilot.png)](https://github.com/ruslanmv/GamePilot "GamePilot")

GamePilot is a generic architecture for AI that plays games, built so that the same underlying system can adapt across different titles rather than being hand-coded for one. Its most distinctive feature is an autopilot mode designed for the long, repetitive stretches of play that games often demand, where the agent can take over mission-assist or away-from-keyboard scenarios. It is as much an exploration of general game-playing agents as it is a practical tool, and it sits at the intersection of reinforcement learning and the multi-agent patterns I use elsewhere.

[Source](https://github.com/ruslanmv/GamePilot)

## Best of the Best

[![Best of the Best](/assets/images/posts/projects/best-of-the-best.png)](https://ruslanmv.com/Best-of-the-Best/ "Best of the Best")

Best of the Best is an autonomous system that curates the AI ecosystem so I do not have to. Every day it monitors the places where new work appears, ranks what it finds, and surfaces the most important models, repositories, and papers, all without my involvement. It is a small, self-contained demonstration of the alive-system philosophy behind Agent-Matrix, an agent that quietly does a real job on a schedule, and it doubles as my own daily window into what is moving in the field.

[Explore](https://ruslanmv.com/Best-of-the-Best/)

## AutoSelf

AutoSelf is a research project on autonomous robotics built on ROS 2, organised around a loop in which the system executes an action, verifies the outcome, and corrects itself when reality diverges from the plan. It explores how a multi-agent approach can make robots reliable enough to operate where no human can intervene, including the demanding context of autonomous construction in extreme environments. It is ongoing collaborative research, so the description here is intentionally high-level; a fuller account will follow as the work is published.

<!-- NOTE for Ruslan: AutoSelf is co-authored research. Confirm what is yours to share
     publicly before adding links, figures, or detail beyond this high-level summary. -->

---

## Data Analysis at  NUMEN

[![homepage](https://ruslanmv.com/assets/images/NUMEN.jpg)](https://web.infn.it/NUMEN/index.php/it/collaboration "Redirect NUMEN collaboration")

I am part of the experimental [NUMEN](https://web.infn.it/NUMEN/index.php/it/collaboration) collaboration. In the NUMEN project, nuclear reactions of double charge-exchange (DCE) will be used as a tool to extract information on the ββ nuclear matrix elements (NME). Nuclear matrix elements for neutrinoless double beta decay and double charge exchange in the scheme of microscopic Interacting Boson Model are important allowing me belong to the NUMEN collaboration.

My research provides insights to compute accurate nuclear matrix elements that serve to improve the nuclear reactions calculations that can be compared with the current experimental data. By using **machine learning** techniques, such as regression and classification techniques it was possible develop an accurate method to fit parameters to construct the nuclear wave functions for complex nuclei allowing present the results by using plots in Python or R, which helps to understand the complex quantum structure of the nuclei. The implementation of **new technologies** is part of the aim of my future work

## Virial Expansion of Nuclear Equation

[![homepage](https://ruslanmv.com/assets/images/texas.png)](https://cyclotron.tamu.edu/ "Redirect TAMU")

Texas A&M University and Cyclotron Institute
College Station, Texas (United States).
Programming Language: Python with Mathematica

I have developed a **model** to study the transition between mater and plasma. This may help to understand the external conditions of nature to materialize the matter. This provides insight into how was created our universe in terms of thermodynamics. I have proposed a function in terms of energy and with my team, we expanded the energy per particle (E/A) of symmetric infinite nuclear matter in powers of the density to take into account 2, 3, . . . , N-body forces.

Here we have used a **Machine Learning** technique of **Regression** in Python and used it in Mathematica to check our results.

The new model is proposed by fitting ground state properties of nuclear matter (binding energy, compressibility, and pressure) and assuming that at high densities a second-order phase transition to the quark-gluon plasma (QGP) occurs.

This work helped to understand the nature of the nuclear matter and research more interesting properties of the plasma such are the sun and how under certain conditions the matter if created.

![](https://github.com/ruslanmv/ruslanmv.github.io/raw/master/assets/images/image%201.jpg)

This work was published in [World Scientifc](https://www.worldscientific.com/doi/abs/10.1142/S0218301312500061) as: Int.J.Mod.Phys. E21 (2012) 1250006

## Classification of the nuclei by knowing their spectrum and perform analysis of nuclear reactions

[![homepage](https://ruslanmv.com/assets/images/infn.jpg)](https://www.ge.infn.it/wordpress/ "Redirect INFN Genova")

National Institute for Nuclear Physics
Genova (Italy)
Programming Language: Python with Fortran

In this project, my task was to provide accurate wavefunctions of the 64Ni and 66Ni. To do that I required to compute the best parameters of the model which reproduce the experimental data coming from the [Database](https://www.nndc.bnl.gov/nudat2/) Nndc - Brookhaven National Laboratory. With this data, I have created a data set of all even-even nuclei which we have labeled the parameters. The parameters of the 66Ni and 66Ni were obtained by using a classification program in Python and by using the results I could determine the spectrum which are helpful to perform the analysis. The model is written in Fortran, but I was able to use python to get the coefficients. This work helped to study short-range correlations of sequential and direct reactions.

![](https://github.com/ruslanmv/ruslanmv.github.io/raw/master/assets/images/image%202.jpg)

This work was published [Physical Review](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.96.044612) C 96, 044612 (2017)

## Analysis and development of a model of the nuclear reaction of Double Charge Exchange

[![homepage](https://ruslanmv.com/assets/images/infn.jpg)](https://www.ge.infn.it/wordpress/ "Redirect INFN Genova")

National Institute for Nuclear Physics
Genova (Italy)
Programming Language: Python, Wolfram Mathematica, Matlab, C

In this project, I have proposed a model to try to explain a nuclear reaction of a double charge exchange that allows determining the nature of the neutrinos. Find constraints to determine if the neutrinos are antiparticles and particles at the same time. My main role was to give the idea of the process and perform the calculation.

I have performed the calculation by using sophisticated numeric and algebraic methods. We found correlations between neutrinoless double beta decay and double charge exchange and this opens the nuclear physicist community put constraints in the research of the masses of the neutrinos and explorer the new physics.

![](https://github.com/ruslanmv/ruslanmv.github.io/raw/master/assets/images/image%204.jpg)

![](https://github.com/ruslanmv/ruslanmv.github.io/raw/master/assets/images/image5.jpg)

This work was published in [Physical Review](https://journals.aps.org/prc/abstract/10.1103/PhysRevC.98.061601) Phys. Rev. C 98, 061601(R)

## HPC CUDA Developments

![](https://github.com/ruslanmv/ruslanmv.github.io/raw/master/assets/images/nvdia.png)

I have developed software in CUDA language that used in clusters to perform heavy calculations.

## Developing of CUDA software for the analysis of flooding of a geographical area.

[![homepage](https://ruslanmv.com/assets/images/reca.jpg)](https://www.redrisk.com/ "Redirect REC")

RED Risk Engineering + Development
Pavia (Italy)
Programming Language,C,C++ ,Fortran, CUDA

In this project, the Italian Association of Insurers enquired RED about some flood model results previously obtained for Italy and the South East of Europe in a different project. RED carried out an exhaustive analysis to evaluate the consistency of the flood event generation process, in the case of both defended and undefended areas, thus taking into account the asset vulnerability. Methodological and technical improvements were suggested to obtain a more accurate representation of the flood hazard in Italy, the results of the G-Cat model were analyzed for the 2014 flood of Modena (Italy), for residential buildings, using an exposure dataset. My role in this project was to help the software developers the porting of the code to CUDA and setup the GPUs ready to the servers for debugging on-premises servers. This program is useful to determine the risk of flooding for some geographical areas.

![](https://github.com/ruslanmv/ruslanmv.github.io/raw/master/assets/images/rec.jpg)

The code is private and belong to the [RED Risk Engineering](https://www.redrisk.com/)

<!-- ECOSYSTEM-JSONLD -->
<script type="application/ld+json">
{
  "@context": "https://schema.org",
  "@type": "CollectionPage",
  "name": "Projects - Ruslan Magana Vsevolodovna",
  "url": "https://ruslanmv.com/projects",
  "author": {
    "@type": "Person",
    "name": "Ruslan Magaña Vsevolodovna",
    "url": "https://ruslanmv.com"
  },
  "mainEntity": {
    "@type": "ItemList",
    "itemListElement": [
      { "@type": "ListItem", "position": 1, "name": "Agent-Matrix", "url": "https://agent-matrix.github.io" },
      { "@type": "ListItem", "position": 2, "name": "MatrixHub", "url": "https://www.matrixhub.io" },
      { "@type": "ListItem", "position": 3, "name": "OllaBridge", "url": "https://ollabridge.com" },
      { "@type": "ListItem", "position": 4, "name": "MedOS", "url": "https://ai-medical-chabot.com" },
      { "@type": "ListItem", "position": 5, "name": "LearnAI", "url": "https://learnskillsai.com" },
      { "@type": "ListItem", "position": 6, "name": "GitPilot", "url": "https://github.com/ruslanmv/gitpilot" },
      { "@type": "ListItem", "position": 7, "name": "HomePilot", "url": "https://yourfriend.online" },
      { "@type": "ListItem", "position": 8, "name": "VRSecretary", "url": "https://github.com/ruslanmv/VRSecretary" },
      { "@type": "ListItem", "position": 9, "name": "GamePilot", "url": "https://github.com/ruslanmv/GamePilot" },
      { "@type": "ListItem", "position": 10, "name": "Best of the Best", "url": "https://ruslanmv.com/Best-of-the-Best/" },
      { "@type": "ListItem", "position": 11, "name": "AutoSelf", "url": "https://ruslanmv.com/projects#autoself" }
    ]
  }
}
</script>
