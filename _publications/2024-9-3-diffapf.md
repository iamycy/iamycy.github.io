---
title: "Differentiable All-pole Filters for Time-varying Audio Systems"
collection: publications
permalink: /publications/2024-9-3-diffapf
excerpt:
date: 2024-9-3
venue: 'International Conference on Digital Audio Effects (DAFx)'
paperurl: 'https://github.com/IoSR-Surrey/DAFx24-Proceedings/raw/main/papers/DAFx24_paper_75.pdf'
citation: 'Chin-Yun Yu, Christopher Mitcheltree, Alistair Carson, Stefan Bilbao, Joshua Reiss and György Fazekas, &quot;Differentiable All-pole Filters for Time-varying Audio Systems&quot;, <i>International Conference on Digital Audio Effects</i>, September 2024.'
---
Infinite impulse response filters are an essential building block of many time-varying audio systems, such as audio effects and synthesisers. However, their recursive structure impedes end-to-end training of these systems using automatic differentiation. Although non-recursive filter approximations like frequency sampling and frame-based processing have been proposed and widely used in previous works, they cannot accurately reflect the gradient of the original system. We alleviate this difficulty by re-expressing a time-varying all-pole filter to backpropagate the gradients through itself, so the filter implementation is not bound to the technical limitations of automatic differentiation frameworks. This implementation can be employed within audio systems containing filters with poles for efficient gradient evaluation. We demonstrate its training efficiency and expressive capabilities for modelling real-world dynamic audio systems on a phaser, time-varying subtractive synthesiser, and feed-forward compressor. We make our code and audio samples available and provide the trained audio effect and synth models in a VST plugin at [this URL](https://diffapf.github.io/web/).