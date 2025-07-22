---
layout: page
title: Audio-Visual Speech Separation via Bottleneck Iterative Network 
img: assets/img/avssbin/MultimodalFusion4MultimediaAnalysis.png
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

<img src="/assets/img/avssbin/avssmotive.png" width="800"/>

A traditional fusion method focuses on learning representations from unimodal inputs and fusion spaces (blue arrows in the plot), but lacks bringing the fusion and late representation information back to the early embedding space (black arrows with the question marks in the plot). 

This backward connection can be important. Consider an audio-visual speech separation (AVSS) task where For example, a female speaker says "WE MUST ADOPT THAT WAY OF" while the other
female speaker says "THEY’RE LIKE THE BASEBALL SCOUTS 20 YEARS", but because of the distortion from the noise and the tones of the two speakers, the two phrases sound close to each other at the beginning words "WE MUST" and "THEY’RE". With no backward connection, the required  information at the early stage might be lost or compressed too much to be recovered correctly in the final fused latent representations. Meanwhile, a backward connection adds back the fused late latent representation to the original embeddings as residuals, then process the combined reprenstation again to get a refined used latent representation, so the early information is constantly added to avoid information loss.

<img src="/assets/img/avssbin/avssmotive2.png" width="800"/>


Paper Citation
==============

```
@misc{zhang2025audiovisualspeechseparationbottleneck,
      title={Audio-Visual Speech Separation via Bottleneck Iterative Network}, 
      author={Sidong Zhang and Shiv Shankar and Trang Nguyen and Andrea Fanelli and Madalina Fiterau},
      year={2025},
      eprint={2507.07270},
      archivePrefix={arXiv},
      primaryClass={cs.SD},
      url={https://arxiv.org/abs/2507.07270}, 
}
```


Audio Samples
=============

<!-- <div class="container">
  <div class="row">
    <div class="col-sm mt-3 mt-md-0">
         Audio Mixture
    </div>
    <div class="col-sm mt-3 mt-md-0">
         Ground Truth Text
    </div>
    <div class="col-sm mt-3 mt-md-0">
         Clean Audio
    </div>
    <div class="col-sm mt-3 mt-md-0">
         AVLIT Output
    </div>
    <div class="col-sm mt-3 mt-md-0">
         BIN Output
    </div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
         {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/mixture_noisy.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
         WE MUST ADOPT THAT WAY OF <br> THEY’RE LIKE THE BASEBALL SCOUTS 20 YEARS
    </div>
    <div class="col-sm mt-3 mt-md-0">
         {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/clean_spk1.wav" controls=true %} <br> {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/clean_spk2.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
         {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/avlit_early_sep_spk1.wav" controls=true %} <br> {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/avlit_early_sep_spk2.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
         {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/profusion_sep_spk1.wav" controls=true %} <br> {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/profusion_sep_spk2.wav" controls=true %}
  	</div>

</div>
 -->

<h3>Case 1: BIN avoids the separation distortion at the begining and in the middle brought by AVLIT and RTFS-Net</h3>



 <div class="container">
  <div class="row">
    <div class="col-sm mt-3 mt-md-0">
        Audio Mixture
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/mixture_noisy.wav" controls=true %}
  	</div>
  	<div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        Ground Truth Text
    </div>
    <div class="col-sm mt-3 mt-md-0">
        WE MUST ADOPT THAT WAY OF
    </div>
    <div class="col-sm mt-3 mt-md-0">
        THEY’RE LIKE THE BASEBALL SCOUTS 20 YEARS
    </div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        Clean Audio
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/clean_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/clean_spk2.wav" controls=true %}
  	</div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        AVLIT Output
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/avlit_early_sep_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/avlit_early_sep_spk2.wav" controls=true %}
  	</div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        RTFS-Net output
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/rtfs_sep_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/rtfs_sep_spk2.wav" controls=true %}
  	</div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        BIN Output
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/profusion_sep_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/WEMUST+THEYRELIKE/profusion_sep_spk2.wav" controls=true %}
  	</div>
  	<div class="w-100"></div>
</div>



<h3>Case 2: BIN separates the audio with some flaws, AVLIT fails to separate the audio, RTFS-Net assigns the separated audio to the wrong person</h3>



 <div class="container">
  <div class="row">
    <div class="col-sm mt-3 mt-md-0">
        Audio Mixture
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/HOWCAN+IMAY/mixture_noisy.wav" controls=true %}
  	</div>
  	<div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        Ground Truth Text
    </div>
    <div class="col-sm mt-3 mt-md-0">
        I MAY NEVER GO TO YOUR HOUSE YOU MAY
    </div>
    <div class="col-sm mt-3 mt-md-0">
        HOW CAN WE GET SO MUCH INFO
    </div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        Clean Audio
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/HOWCAN+IMAY/clean_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/HOWCAN+IMAY/clean_spk2.wav" controls=true %}
  	</div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        AVLIT Output
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/HOWCAN+IMAY/avlit_sep_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/HOWCAN+IMAY/avlit_sep_spk2.wav" controls=true %}
  	</div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        RTFS-Net output
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/HOWCAN+IMAY/rtfs_sep_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/HOWCAN+IMAY/rtfs_sep_spk2.wav" controls=true %}
  	</div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        BIN Output
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/HOWCAN+IMAY/profusion_sep_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/HOWCAN+IMAY/profusion_sep_spk2.wav" controls=true %}
  	</div>
  	<div class="w-100"></div>
</div>




<h3>Case 3: All models fail to separate the audio</h3>



 <div class="container">
  <div class="row">
    <div class="col-sm mt-3 mt-md-0">
        Audio Mixture
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/THISECO+ANDYOU/mixture_noisy.wav" controls=true %}
  	</div>
  	<div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        Ground Truth Text
    </div>
    <div class="col-sm mt-3 mt-md-0">
        THIS ECONOMY WON'T WORK AND
    </div>
    <div class="col-sm mt-3 mt-md-0">
        AND YOU WALK AROUND AND IT POINTS
    </div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        Clean Audio
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/THISECO+ANDYOU/clean_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/THISECO+ANDYOU/clean_spk2.wav" controls=true %}
  	</div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        AVLIT Output
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/THISECO+ANDYOU/avlit_sep_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/THISECO+ANDYOU/avlit_sep_spk2.wav" controls=true %}
  	</div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        RTFS-Net output
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/THISECO+ANDYOU/rtfs_sep_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/THISECO+ANDYOU/rtfs_sep_spk2.wav" controls=true %}
  	</div>
    <div class="w-100"></div>
    <div class="col-sm mt-3 mt-md-0">
        BIN Output
    </div>
    <div class="col-sm mt-3 mt-md-0">
        {% include audio.liquid path="assets/audio/avssbin/THISECO+ANDYOU/profusion_sep_spk1.wav" controls=true %}
  	</div>
  	<div class="col-sm mt-3 mt-md-0">
  		{% include audio.liquid path="assets/audio/avssbin/THISECO+ANDYOU/profusion_sep_spk2.wav" controls=true %}
  	</div>
  	<div class="w-100"></div>
</div>




<!-- | Audio Mixture | Ground Truth Text | Clean Audio | AVLIT Output | BIN Output |
| ------------- | ----------------- | ----------- | ------------ | ---------- |
| {% include audio.liquid path="https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/mixture_noisy.wav" controls=true %} | WE MUST ADOPT THAT WAY OF <br> THEY’RE LIKE THE BASEBALL SCOUTS 20 YEARS | {% include audio.liquid path="http://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/clean_spk1.wav" controls=true %} <br> {% include audio.liquid path="http://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/clean_spk2.wav" controls=true %} | {% include audio.liquid path="https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/avlit_early_sep_spk1.wav" controls=true %} <br> {% include audio.liquid path="https://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/avlit_early_sep_spk2.wav" controls=true %}| {% include audio.liquid path="http://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/profusion_sep_spk1.wav" controls=true %} <br> {% include audio.liquid path="http://github.com/Information-Fusion-Lab-Umass/BottleneckIterativeNetwork/blob/main/sample_audio/WEMUST%2BTHEYRELIKE/profusion_sep_spk2.wav" controls=true %} |
 -->

<!-- <img src="/assets/img/adforecasting/mri_degeneration.png" width="800"/> -->

