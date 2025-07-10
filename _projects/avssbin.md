---
layout: page
title: Audio-Visual Speech Separation via Bottleneck Iterative Network 
img: assets/img/adforecasting/MultimodalFusion4MultimediaAnalysis.png
importance: 1
category: academic
bibliography:
- /assets/ref/avssbin.bib
---


Introduction
============


We introduce a new AVSS model, Bottleneck Iterative Network (BIN), that iteratively refines the audio and visual representations using their fused representation via a repetitive progression through the bottleneck fusion variables and the outputs of the two modalities from the same fusion block. Tested on two popular AVSS benchmarks, BIN strikes a
good balance between speech separation quality and computing resources, being on par with RTFS-Net’s state-of-theart performance (and improving on SI-SDR) while saving
up to 74% training time and 80% GPU inference time. Our code is available on [Github](https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork).


Intuition
=======

<img src="/assets/img/avssbing/avssmotive.pdf" width="800"/>

A traditional fusion method focuses on learning representations from unimodal inputs and fusion spaces (blue arrows in the plot), but lacks bringing the fusion and late representation information back to the early embedding space (black arrows with the question marks in the plot). 

This backward connection can be important. Consider an audio-visual speech separation (AVSS) task where For example, a female speaker says "WE MUST ADOPT THAT WAY OF" while the other
female speaker says "THEY’RE LIKE THE BASEBALL SCOUTS 20 YEARS", but because of the distortion from the noise and the tones of the two speakers, the two phrases sound close to each other at the beginning words "WE MUST" and "THEY’RE". With no backward connection, the required  information at the early stage might be lost or compressed too much to be recovered correctly in the final fused latent representations. Meanwhile, a backward connection adds back the fused late latent representation to the original embeddings as residuals, then process the combined reprenstation again to get a refined used latent representation, so the early information is constantly added to avoid information loss.

<img src="/assets/img/avssbing/avssmotive2.pdf" width="800"/>


Audio Samples
=============


| Audio Mixture | Ground Truth Text | Clean Audio | AVLIT Output | BIN Output |
| ------------- | ----------------- | ----------- | ------------ | ---------- |
| ![Audio](https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/mixture_noisy.wav) | WE MUST ADOPT THAT WAY OF <br> THEY’RE LIKE THE BASEBALL SCOUTS 20 YEARS | ![Audio](https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/clean_spk1.wav) <br> ![Audio](https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/clean_spk2.wav) | ![Audio](https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/avlit_early_sep_spk1.wav) <br> ![Audio](https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/avlit_early_sep_spk2.wav) | ![Audio](hhttps://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/profusion_sep_spk1.wav) <br> ![Audio](https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/profusion_sep_spk2.wav) | 


<!-- <img src="/assets/img/adforecasting/mri_degeneration.png" width="800"/> -->

