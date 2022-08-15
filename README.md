# PromptSum
Cool prompting for parameter-efficient few-shot controllable summarization!

# Bash scripts

!!! Don't forget to change the "root" variable at the top of args in main.py!!!

Run the corresponding bash script for each use case:

### Pre-training 
bash run_pretraining.sh

### 0-shot summarization (3 seeds in validation, 1 seed in test)
bash runall_zeroshot_promptsum.sh

### few-shot summarization (3 seeds)
For the <ins>baselines</ins>:

bash runall_kshot_baselines.sh

For the <ins>oracle</ins>:

bash runall_kshot_oracle.sh

For <ins>PromptSum</ins>:

bash runall_kshot_promptsum.sh

For <ins>controllability</ins> experiments:

bash runall_kshot_controllability.sh

For <ins>counterfactual</ins> training experiments:

bash runall_kshot_counterfactual.sh

For <ins>hallucinations</ins> experiments:

bash runall_kshot_hallucination.sh



## Contents
- [PromptSumm](#PromptSumm)
  - [Methodology](#Method)
  - [Related Papers](#related-papers)
    - [Prompting](#prompting)
    - [Summarization](#summarization)

## Method
![Image of first discussion](https://github.com/ntunlp/PromptSumm/blob/main/images/first_discussion_screenshot.jpeg)

## Related Papers

### Prompting
1. **Pre-train, Prompt, and Predict: A Systematic Survey of Prompting Methods in Natural Language Processing.**  Preprint.

   *Liu, Pengfei, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig.*  [[pdf](https://arxiv.org/pdf/2107.13586)] [[project](http://pretrain.nlpedia.ai)], 2021.7
2. **The Power of Scale for Parameter-Efﬁcient Prompt Tuning.** Preprint. ![](https://img.shields.io/badge/Continuous-red) ![](https://img.shields.io/badge/Classification-blue)
   
   *Brian Lester, Rami Al-Rfou, Noah Constant*. [[pdf](https://arxiv.org/pdf/2104.08691.pdf)], [[implementation](https://github.com/kipgparker/soft-prompt-tuning)], 2021.4
   
### Summarization

#### Controlled & guided summarization
1. **Controllable Abstractive Summarization.**  WMT 2018.

   *Angela Fan, David Grangier, Michael Auli.*  [[pdf](https://arxiv.org/pdf/1711.05217.pdf)] [[project]()], 2020.5
2. **CTRLSUM: Towards generic controllable text summarization.**  Preprint.

   *Junxian He, Wojciech Kryściński, Bryan McCann, Nazneen Rajani, Caiming Xiong.*  [[pdf](https://arxiv.org/pdf/2012.04281.pdf)] [[project](https://github.com/salesforce/ctrl-sum)], 2020.12
3. **GSum: A General Framework for Guided Neural Abstractive Summarization.**  NAACL 2021.

   *Zi-Yi Dou, Pengfei Liu, Hiroaki Hayashi, Zhengbao Jiang, Graham Neubig.*  [[pdf](https://arxiv.org/pdf/2010.08014.pdf)] [[project](https://github.com/
neulab/guided_summarization.)], 2021.4

#### Unsupervised summarization
1. **Pegasus: pre-training with extracted gap-sentences for abstractive summarization.**  ICML 2020.

   *Jingqing Zhang, Yao Zhao, Mohammad Saleh, Peter J. Liu.*  [[pdf](https://arxiv.org/pdf/1912.08777.pdf)] [[project](https://github.com/google-research/pegasus)], 2020.7
2. **TED: A Pretrained Unsupervised Summarization Model with Theme Modeling and Denoising.**  EMNLP 2020.

   *Ziyi Yang, Chenguang Zhu, Robert Gmyr, Michael Zeng, Xuedong Huang, Eric Darve.*  [[pdf](https://arxiv.org/pdf/2001.00725.pdf)] [[project]()], 2020.10

