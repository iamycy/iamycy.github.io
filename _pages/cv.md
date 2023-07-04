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
* B.S. in Computer Science, National Chiao Tung University, 2018
* Ph.D in Electronic Engineering and Computer Science, Queen Mary University of London, 2026 (expected)

Work experience
======
* Summer 2017: Intern
  * Institute of Information Science, Academia Sinica
  * Duties included: Developing Cepstrum-Based Music Transcription System
  * Supervisor: Professor Li Su

* 2018: Research Assistant
  * Institute of Information Science, Academia Sinica
  * Duties included: Doing Research on NN-based Vocoder Model/Developing End-to-End Music Transcription Model
  * Supervisor: Professor Li Su

* Summer 2019 - Winter 2020: Engineer
  * Vive R&D, HTC
  * Duties included: Applying Deep Learning Techniques on HRTF applications
  * Supervisor: VP Vasco Choi
  
* Winter 2020 - Summer 2022: Backend Engineer
  * Backend R&D, Rayark Inc.
  * Duties included: Design and implement mobile game servers
  * Supervisor: CTO Alvin


Skills
======
* Programming Language
  * C/C++
  * Python
  * Golang
* Music Information Retrieval
  * Multi Pitch Estimation
  * Source Separation
  * Voice Generation/Synthesis
* Machine Learning
* Deep Learning
* DSP
* Spatial Audio
  * Head Related Transfer Function Spatial Upsampling
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