---
title: "Audio Pipeline Smoke Test"
permalink: /audio-smoke-test/
layout: single
author_profile: false
sitemap: false
read_time: "1 min read"
date: 2026-06-06

# --- essay audio narration (▷ Audio) ---
slug: audio-smoke-test
audio_slug: audio-smoke-test
audio: true
---

This page exists to verify the audio pipeline end to end: that it **builds**,
that the generator **ingests** the tagged text, and that the manifest is
**retrieved** to render the inline ▷ Audio control.

<!-- audio:start -->

This is a short narration sample for the audio pipeline smoke test. If you can
hear this sentence, the text was ingested between the audio tags, narrated by
the configured speaker, uploaded to object storage, and retrieved from the
audio manifest.

<!-- audio:skip:start -->
This sentence sits inside a skip block, so it appears on the page but is never
read aloud.
<!-- audio:skip:end -->

<!-- audio:pause -->

That pause marks the end of the spoken sample. Everything after the closing tag
is page content only.

<!-- audio:end -->

Anything below this line is outside the audio tags and is never narrated — it is
here only to confirm that tagged narration scopes correctly.
