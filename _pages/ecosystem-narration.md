---
# ---------------------------------------------------------------------------
# Narration script for /ecosystem/ — the source of truth for the guided tour.
#
# This file is NOT a published page (`published: false`). It exists so that
# scripts/generate_audio.py can render ecosystem.mp3, upload it to R2, and
# record where every chapter begins.
#
# The contract with the page:
#   <!-- audio:cue: <id> | <title> -->  here
#   data-tour-chapter="<id>"            on the matching <section> in
#                                       _pages/ecosystem.html
#
# Every cue id below must exist on the page, in the same order. The guided-tour
# player logs a console warning for any cue it cannot find.
#
# After editing this script:  make audio-force   (or run the workflow with
# force=true) — narration is only regenerated on purpose, never automatically.
# ---------------------------------------------------------------------------
title: "My Personal AI Ecosystem"
eyebrow: "Personal AI Ecosystem"
headline: "I wanted a personal AI."
subtitle: "So I started building the stack."
author_name: "Ruslan Magana Vsevolodovna"
author_role: "Physicist · Machine Learning Engineer · Genoa, Italy"
thesis: "One AI identity, one memory architecture, and different surfaces for work, meetings, building, and life."
audio: true
audio_slug: ecosystem
published: false
sitemap: false
search: false
---

<!-- audio:start -->

<!-- audio:cue: hero | I wanted a personal AI -->

This is a guided tour of the ecosystem I am building. Fourteen short chapters.
The page will follow along as I talk, so you can put it down and just listen.

Not another chatbot. Not another collection of subscriptions. I wanted an AI
that could know me, remember, work beside me, attend meetings, help build
software, use compute I control, and eventually share both work and life with
me.

<!-- audio:cue: why | The problem is fragmentation -->

The problem is not a lack of AI. It is fragmentation.

Chat, meeting notes, coding, planning, calendars, knowledge, voice, avatars,
and model access usually live in separate products. Each company owns another
slice of my context, charges another subscription, and stores another piece of
my identity.

I want something different. Instead of isolated assistants, one AI identity.
Instead of scattered context, one memory architecture. Instead of a single
provider, compute I can route and control. Instead of one rigid interface,
several places to meet the same AI.

The pieces already existed. The difficult part was making them behave like one
system.

<!-- audio:cue: architecture | Five projects. One personal stack -->

So the architecture is five projects and one stack.

HomePilot holds the center: identity, memory, knowledge, policy, and
orchestration. DayPilot gives that intelligence a workplace. The 3D Avatar
gives it a presence. OllaBridge gives it compute. And GitPilot gives it a way
to help build itself, with me in control.

Five parts, one shared intelligence underneath.

<!-- audio:cue: surfaces | Different places to meet it -->

One intelligence. Different places to meet it.

I do not want a DayPilot AI, an Avatar AI, a meeting AI, and a coding AI that
behave like strangers to each other. I want the same underlying identity and
the same continuity, wherever the conversation happens.

At work, DayPilot organizes the professional workspace. In meetings,
MeetingSense brings the context back into HomePilot. While coding, GitPilot
turns intent into reviewed changes. At home, the Avatar gives the same persona
a presence. And on any machine, OllaBridge decides where the inference actually
runs.

<!-- audio:cue: day | A day with the system -->

Here is what a day with that system looks like. This part is the vision, not a
release claim.

At half past seven, HomePilot brings the priorities and the context, and
DayPilot presents the workday. At nine, deep work: GitPilot helps implement a
feature, using a model routed through OllaBridge. At eleven, a meeting, where
MeetingSense is designed to listen, transcribe, understand the shared material,
and track the actions.

Just after noon, the follow-up. HomePilot could surface three decisions, two
actions, and one unresolved question, and then ask permission before creating
anything.

At half past six, work ends. DayPilot goes quiet and the professional tools
recede. At eight in the evening, the same persona appears through the 3D
Avatar, for voice, cooking, language practice, exercise, a movie, or just
conversation.

The interface changes. The relationship and the memory do not.

<!-- audio:cue: secretary | From chatbot to personal secretary -->

Which brings me to the long-term direction: from chatbot to personal secretary.

The goal is not a model that only waits for prompts. It is an AI that can know
what is happening, remember what matters, prepare me, work with me, and follow
up, while asking before any consequential action.

In the morning it might say: good morning, you have three meetings today, the
pricing proposal from last week is still open, I prepared the previous
decisions and the latest document, would you like a five-minute briefing.

And afterwards: three decisions were made, you own two actions, shall I create
the tasks and draft the follow-up.

That is the shape of it. Always ending in a question, never in an assumption.

<!-- audio:cue: build | The stack helps build the stack -->

There is a loop inside this that I like a lot. The stack helps build the stack.

I can use my own AI infrastructure to help build my own AI infrastructure. This
is not autonomous recursive self-improvement. It is an engineering loop with an
explicit human checkpoint: an idea, an explorer, a planner, a coder, tests, a
reviewer, my approval, and only then the repository.

Underneath the loop sits the compute: OllaBridge, then a local GPU, then a
workstation, then optionally a provider. Commercial models remain genuinely
useful. They are simply no longer the only path.

<!-- audio:cue: local | Local-first, not cloud-absolutist -->

I should be precise about what local-first means here, because it is easy to
turn into a slogan.

Local where practical. Cloud where useful. Choice always.

Three principles. Control: I decide where models and data run. Portability:
applications depend on open interfaces instead of one vendor. Optionality:
commercial providers are used when they add value, not because the system has
no alternative.

I do not want zero cloud. I want zero forced cloud.

Local-first and open-source by default makes paid AI optional rather than
mandatory. Hardware, electricity, optional APIs, and cloud compute can still
cost money. That is an honest trade, not a free lunch.

<!-- audio:cue: solves | Replace seams with architecture -->

What all of this really does is replace seams with architecture.

A fragmented AI identity becomes one HomePilot persona. Meeting knowledge that
used to disappear becomes MeetingSense and retrieval. Scattered work context
becomes DayPilot. Every application needing its own model endpoint becomes
OllaBridge. AI coding tied to one provider becomes GitPilot. An AI with no
presence gets a 3D Avatar. Multiplying subscriptions become local and open
alternatives where that is practical. And private context split across vendors
becomes self-hosting with explicit permissions.

<!-- audio:cue: projects | Independent today. Designed to converge -->

Now the projects themselves. They are independent today, and designed to
converge.

HomePilot is who my AI is: the personal backend for personas, identity, memory,
knowledge, tools, communication, policy, and model access.

DayPilot is how my AI works with me: a professional cockpit where those
personas can operate with context and with approval.

The 3D Avatar is how my AI shares space with me: embodiment, voice, expression,
and eventually a social and leisure surface for the same persona.

OllaBridge is where my AI thinks: one open, compatible control plane between
the applications and the place inference actually runs.

And GitPilot helps me build the ecosystem itself: specialized agents that
explore, plan, generate, test, and review, with approval modes and repository
integration.

These are active repositories at different stages. Some of what I just
described is running today. Some of it is direction.

<!-- audio:cue: manifesto | The manifesto -->

If I compress all of it into four lines, it is this. Own the AI. Own the
memory. Own the compute. Own the tools.

My AI should remember me because I choose it to. My data should not require a
software subscription to exist. My applications should survive a change of
model provider. My GPU should be useful when I already own one. An AI should
ask before taking consequential actions. Work AI and personal AI should not
require separate identities. Open source should make the system inspectable and
adaptable. Local-first does not mean isolated. And automation should increase
my control, not remove it.

<!-- audio:cue: roadmap | Capability stages, not release promises -->

Where is this going? In capability stages, not release promises.

Now: the core applications exist independently and keep evolving. Connecting:
personas, model routing, meeting intelligence, and tools converge. Working
together: DayPilot and HomePilot form one continuous work system. Present: the
Avatar gives the same intelligence an embodied surface. And finally ambient,
where the AI becomes reachable across devices while still respecting
boundaries.

<!-- audio:cue: open | Built for myself. Open so others can inspect -->

I am building this in public.

It is a personal engineering project and a long-term vision, not a pitch deck.
The repositories are public so that people can learn from them, fork them,
improve them, or use only the pieces they need.

<!-- audio:cue: final | The reason -->

So, the reason.

I am not trying to build another chatbot. I am trying to build the AI I wanted
to have. One that knows my work. Remembers what matters. Runs on infrastructure
I can control. Helps build the software around it. And eventually shares both
the productive and the ordinary moments with me.

Thank you for listening. The links to every repository are on this page.

<!-- audio:end -->
