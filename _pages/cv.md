---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* B.S. in Computer Science, National Chiao Tung University, Taiwan, 2018
* Ph.D in Electronic Engineering and Computer Science, Queen Mary University of London, UK, 2026 (expected)

Work experience
======
* Summer 2017: Intern
  * Institute of Information Science, Academia Sinica, Taipei, Taiwan
  * Duties include: Developing Cepstrum-Based Music Transcription System
  * Supervisor: Professor Li Su

* 2018: Research Assistant
  * Institute of Information Science, Academia Sinica, Taipei, Taiwan
  * Duties include: Doing Research on NN-based Vocoder Model/Developing End-to-End Music Transcription Model
  * Supervisor: Professor Li Su

* Summer 2019 - Winter 2020: Engineer
  * Vive R&D, HTC, New Taipei City, Taiwan
  * Duties include: Applying Deep Learning Techniques on HRTF applications
  * Supervisor: VP Vasco Choi
  
* Winter 2020 - Summer 2022: Backend Engineer
  * Backend R&D, Rayark Inc., Taipei, Taiwan
  * Duties include: Design and implement mobile game servers
  * Supervisor: CTO Alvin

* Summer 2024 - Present: Research Intern
  * Sony AI, Sony, Tokyo, Japan
  * Duties include: differentiable audio processing and audio mixing
  * Supervisor: Manager Weih Siang Liao


Skills
======
* Programming Language
  * C/C++
  * Python
  * Golang
  * Haskell
* Music Information Retrieval
  * Multi-Pitch Estimation
  * Source Separation
  * Voice Generation/Synthesis
* Machine Learning
* Deep Learning
* DSP
* Spatial Audio
  * Head Related Transfer Functions
* Qiskit (Quantum Programming)
* Music Production
  * Song Writing
  * Drums Editing
  * Mixing
* Musical Instruments
  * Piano
  * Violin
  * Electric Guitar

Publications
======
  <ul>{% for post in site.publications reversed %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
  
Talks
======
  <ul>{% for post in site.talks %}
    {% include archive-single-talk-cv.html %}
  {% endfor %}</ul>
