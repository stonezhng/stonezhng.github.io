---
layout: page
title: A Deep Learning Approach to Early Alzheimer’s Forecasting in the Wild
img: assets/img/adforecasting/titleimg.png
importance: 1
category: academic
bibliography:
- /assets/ref/adforecasting.bib
---


Introduction
============

Alzhiemer's Disease (AD) is a brain degenerative disease which causes
60-70% of cases of dementia [@pmid2397368] and affects an estimated 6.7
million people age 65 and older [@pmid36918389]. Drug and non-drug
treatments have been conducted symptomatically against dementia in the
late stage of AD [@pmid36918389] [@VAZ2020173554], but they are
inefficacious to modify the progression of AD [@VAZ2020173554]. On the
other hand, disease-modifying therapies (DMT) focusing on interventions
based on the amyloid cascade hypothesis and tau biology [@pmid30501965]
are able to preserve cognitive and functional capacity of the patients,
but they require accurate diagnosis at the pre-dementia and preclinical
stages [@pmid31994640], the identification of which is still a
challenging topic in the neuroscience community [@pmid23183885]. Since
recent interdisciplinary studies demonstrate successful application of
machine learning analysis to medical data including images
[@pmid28301734] and time series [@pmid33693379], we propose a deep
learning approach to analysis subjects' longitudinal records and
determine the subjects' risk in developing AD over time. The risk
factors generated by our model then suggest whether subjects will
develop AD in the future, serving as indicators of clinical trail
recruitment and DMT early application.

Significance
============

#### Clinical medicine

<!--
![image](/assets/img/adforecasting/mri_degeneration.png){ width: 200px; }-->

<img src="/assets/img/adforecasting/mri_degeneration.png" width="800"/>

At present, no method -- mathematical, clinical, or epidemiological --
predicts the development of AD, limiting the identification of cohorts
to conduct early symptomatic treatments and DMT. Since the majority of
AD treatments happens in the late stage [@VAZ2020173554][@pmid35286590],
the damage caused by the disease to the brain is generally irreversible.
For example, without early AD diagnosis and early intervention,
amyloid-$\beta$ cascade often has already produced cognitive symptoms
before treatments are applied, resulting in limited success in
pharmacotherapies targeting amyloid-$\beta$ and tau in the late stage of
the disease [@pmid33268824]. Also, degenerative pathology cased by
late-onset AD is observed on hippocampal volume and cortical thickness
in several brain regions computed from MRIs [@pmid30217936] (figure
[\[fig:mridegen\]](#fig:mridegen){reference-type="ref"
reference="fig:mridegen"}). To assist early AD diagnosis, our deep
learning model will be able to predict future AD stages from the
observed subjects' medical records and allow 1) the identification of
patient cohorts for potential recruitment of clinical trials; and 2) the
testing of early interventions and medications that interrupt the course
of AD.

#### Neuroscience

There are keen efforts in neuroscience research on the early detection
of AD. Some of the well studied key risks include mild cognitive
impairment (MCI) subtype (e.g., aMCI, naMCI), poor performance on
various neurocognitive tests, and biomarkers such as abnormal
cerebrospinal fluid (CSF) tau or tau/amyloid-$\beta$ ratio, APOE4
positive status, white matter hyperintensities, and atrophy in the
hippocampal, medial temporal, or entorhinal regions [@pmid35286590]
[@pmid30284855] [@pmid30320579] [@pmid31682146]. To analyse numerous
risk factors efficiently, data-driven AD progression models were
proposed and revealed multifactorial interactions between various risk
factors and disease progressions [@pmid30321505]. We take a further step
towards this interdisciplinary approach by utilizing the high capacity
of deep learning models to 1) process high volumes of data of multiple
modalities including cognitive test scores, demograpchis, biomarkers,
and volumetric features computed from magnetic resonance imaging (MRI)
on regions of interest; 2) introduce full MRI modality besides MRI based
engineered features via computer vision and information theory
techniques; and 3) capture future health states transitions and mapping
from health states abstraction to a diagnosis as an assistance to
neuroscience research.

Related Deep Learning Work
==============================

Recently, applying deep learning models to AD studies has been
addressing increasing interests. From a computer vision perspective, 2D
[@Valliani] [@JAIN2019147] or 3D [@liu2020design] convolutional neural
networks (CNN) are applied to MRI 2D slices or full MRIs to extract
latent representations for AD classification. People also study AD
future progress detection and forecasting via time series models such as
recurrent neural networks (RNN) [@ALBRIGHT2019483] [@flare],
bidirectional-RNN [@RAHIM2023363] and long short-term memory networks
(LSTM) [@ABUHMED2021106688], reaching a maximal forecasting window of
2.5 years.

Potential Translational Impact
==============================

The proposed deep learning model that forecasts future AD stages can be
naturally applied to multiple real-world scenarios. a) By integrating
the model to electronic health record management systems, the model can
perform AD stages forecasting per patient utilizing the historical
records stored in the system in a regular time interval, say 6 months,
and will be able to raise early alerts once future dementia is detected.
b) The model can be embedded into MRI machines and do real-time analysis
once MRIs are captured. c) When pharma companies or neuroscience
researchers identify individuals to be recruited for clinical trails,
our model can serve as an assistance software that takes available
individual's records and outputs the estimated future AD progression, so
that clinical trails can recruit more patients with light or no
cognitive issues but progressing pathological changes. In summary, upon
completion, this project will make it possible to inform patients of
their risks in a timely manner and select subjects for clinical trials.
