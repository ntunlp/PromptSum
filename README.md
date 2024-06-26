# PromptSum

## Environment

Once you clone the repo, create a dedicated conda environment with Python 3.7: 
```bash
cd PromptSum/
conda create --name promptsum python=3.7
```

Next activate the environment:
```bash
conda activate promptsum
```

Then install all the dependencies:
```bash
pip install -r requirements.txt
```

## Experiments

First change the global variables in src/hyperparameters.py according to your respective local paths. 

To run **pre-training**:
```
bash src/scripts/run_pretraining.sh
```

If you cannot do the pretraining, you can use our checkpoint [here](https://drive.google.com/file/d/1Cc1RYHIdRv_0S6JJNm4-Drwh2YA4DiiH/view?usp=sharing).  
After unzipping the folder, place it in *pretrained_ckpt/*.

To run **0-shot** summarization (3 seeds in validation, 1 seed in test):
```
bash src/scripts/run_zeroshot.sh
```

To run **few-shot** summarization (3 seeds):
```
bash src/scripts/run_kshot_promptsum.sh
bash src/scripts/run_kshot_controllability.sh
bash src/scripts/run_kshot_counterfactual.sh
bash src/scripts/run_kshot_hallucination.sh
```

To run **full-shot** summarization (1 seed):
```
bash src/scripts/run_fullshot_promptsum.sh
```

## Citation

If you find any of this useful, please kindly consider citing our paper in your publication.

```
@article{ravaut2023promptsum,
  title={Promptsum: Parameter-efficient controllable abstractive summarization},
  author={Ravaut, Mathieu and Chen, Hailin and Zhao, Ruochen and Qin, Chengwei and Joty, Shafiq and Chen, Nancy},
  journal={arXiv preprint arXiv:2308.03117},
  year={2023}
}
```
