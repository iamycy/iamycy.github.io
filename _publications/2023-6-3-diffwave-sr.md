---
title: "Conditioning and Sampling in Variational Diffusion Models for Speech Super-Resolution"
collection: publications
category: conferences
permalink: /publications/2023-6-3-diffwave-sr
excerpt:
date: 2023-6-3
venue: 'IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)'
paperurl: 'https://ieeexplore.ieee.org/abstract/document/10095103'
citation: 'Chin-Yun Yu, Sung-Lin Yeh, György Fazekas, and Hao Tang, &quot;Conditioning and Sampling in Variational Diffusion Models for Speech Super-Resolution&quot;, <i>IEEE International Conference on Acoustics, Speech and Signal Processing</i>, June 2023.'
---
Recently, diffusion models (DMs) have been increasingly used in audio processing tasks, including speech super-resolution (SR), which aims to restore high-frequency content given low-resolution speech utterances. This is commonly achieved by conditioning the network of noise predictor with low-resolution audio. In this paper, we propose a novel sampling algorithm that communicates the information of the low-resolution audio via the reverse sampling process of DMs. The proposed method can be a drop-in replacement for the vanilla sampling process and can significantly improve the performance of the existing works. Moreover, by coupling the proposed sampling method with an unconditional DM, i.e., a DM with no auxiliary inputs to its noise predictor, we can generalize it to a wide range of SR setups. We also attain state-of-the-art results on the VCTK Multi-Speaker benchmark with this novel formulation.