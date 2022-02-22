---
title: "Harmonic Preserving Neural Networks for Efficient and Robust Multipitch Estimation"
collection: publications
permalink: /publications/2020-12-10-harmonic-preserve-network
excerpt:
date: 2020-12-10
venue: '2020 Asia-Pacific Signal and Information Processing Association Annual Summit and Conference (APSIPA ASC)'
paperurl: 'https://ieeexplore.ieee.org/abstract/document/9306345'
citation: 'Chin-Yun Yu, Jing-Hua Lin, and Li Su, &quot;Harmonic Preserving Neural Networks for Efficient and Robust Multipitch Estimation&quot;, <i>Asia-Pacific Signal and Information Processing Association Annual Summit and Conference</i>, December 2020.'
---
Multi-pitch estimation (MPE) is a fundamental yet challenging task in audio processing. Recent MPE techniques based on deep learning have shown improved performance but are computation-hungry and relatively sensitive to the variation of data such as noise contamination, cross-domain data, etc. In this paper, we present the harmonic preserving neural network (HPNN), a model that incorporates deep learning and domain knowledge in signal processing to improve the efficiency and robustness of MPE. The proposed method starts from the multilayered cepstrum (MLC), a feature representation that utilizes repeated Fourier transform and nonlinear scaling to suppress the non-periodic components in signals. Following the combined frequency and periodicity (CFP) principle, the last two layers of the MLC are integrated to suppress the harmonics of pitches in the spectrum and enhance the components of true fundamental frequencies. A convolutional neural network (CNN) is then placed to further optimize the pitch activation. The whole system is constructed as an end-to-end learning scheme. Improved time efficiency and performance robustness to noise and cross-domain data are demonstrated with experiments on polyphonic music in various noise levels and multi-talker speech.