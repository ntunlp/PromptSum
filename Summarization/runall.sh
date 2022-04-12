## 64-shot
## train t5 tagger
echo "start --few_shot 64 --train_t5_tagger --if_spacy"
#python main.py --few_shot 64 --train_t5_tagger --if_spacy
echo "finish --few_shot 64 --train_t5_tagger --if_spacy"
## oracle: target (T5 entity)
echo "start --few_shot 64 --guidance_mode target"
python main.py --few_shot 64 --guidance_mode target
echo "finish --few_shot 64 --guidance_mode target"
## T5 tagger (generate entities in summary)
echo "start --few_shot 64 --guidance_mode input"
python main.py --few_shot 64 --guidance_mode input
echo "finish --few_shot 64 --guidance_mode input"


## 100-shot
## train t5 tagger
echo "start --few_shot 100 --train_t5_tagger --if_spacy"
python main.py --few_shot 100 --train_t5_tagger --if_spacy
echo "finish --few_shot 100 --train_t5_tagger --if_spacy"
## oracle: target (T5 entity)
echo "start --few_shot 100 --guidance_mode target"
#python main.py --few_shot 100 --guidance_mode target
echo "finish --few_shot 100 --guidance_mode target"
## T5 tagger (generate entities in summary)
echo "start --few_shot 100 --guidance_mode input"
python main.py --few_shot 100 --guidance_mode input
echo "finish --few_shot 100 --guidance_mode input"


## 10-shot
## train t5 tagger
echo "start --few_shot 10 --train_t5_tagger --if_spacy"
python main.py --few_shot 10 --train_t5_tagger --if_spacy
echo "finish --few_shot 10 --train_t5_tagger --if_spacy"
## oracle: target (T5 entity)
echo "start --few_shot 10 --guidance_mode target"
#python main.py --few_shot 10 --guidance_mode target
echo "finish --few_shot 10 --guidance_mode target"
## T5 tagger (generate entities in summary)
echo "start --few_shot 10 --guidance_mode input"
python main.py --few_shot 10 --guidance_mode input
echo "finish --few_shot 10 --guidance_mode input"