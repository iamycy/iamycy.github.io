---
title: "DiffVox: A Differentiable Model for Capturing and Analysing Vocal Effects Distributions"
collection: publications
category: conferences
permalink: /publication/2025-09-02-diffvox
excerpt: ""
date: 2025-09-02
venue: 'International Conference on Digital Audio Effects (DAFx)'
paperurl: 'https://www.dafx.de/paper-archive/2025/DAFx25_paper_9.pdf'
citation: 'Chin-Yun Yu, Marco A. Martínez-Ramírez, Junghyun Koo, Ben Hayes, Wei-Hsiang Liao, György Fazekas and Yuki Mitsufuji, &quot;DiffVox: A Differentiable Model for Capturing and Analysing Vocal Effects Distributions&quot;, <i>International Conference on Digital Audio Effects (DAFx)</i>, September 2025.'
---

This study introduces a novel and interpretable model, DiffVox, for matching vocal effects in music production. DiffVox, short for "Differentiable Vocal Fx", integrates parametric equalisation, dynamic range control, delay, and reverb with efficient differentiable implementations to enable gradient-based optimisation for parameter estimation. Vocal presets are retrieved from two datasets, comprising 70 tracks from MedleyDB and 365 tracks from a private collection. Analysis of parameter correlations reveals strong relationships between effects and parameters, such as the high-pass and low-shelf filters often working together to shape the low end, and the delay time correlating with the intensity of the delayed signals. Principal component analysis reveals connections to McAdams' timbre dimensions, where the most crucial component modulates the perceived spaciousness while the secondary components influence spectral brightness. Statistical testing confirms the non-Gaussian nature of the parameter distribution, highlighting the complexity of the vocal effects space. These initial findings on the parameter distributions set the foundation for future research in vocal effects modelling and automatic mixing. Our source code and datasets are accessible at [this URL](https://github.com/SonyResearch/diffvox).
