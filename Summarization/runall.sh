## 64-shot
echo "start --few_shot 64 --train_t5_tagger --if_spacy"
#python main.py --few_shot 64 --train_t5_tagger --if_spacy
echo "finish --few_shot 64 --train_t5_tagger --if_spacy"

echo "start --few_shot 64 --guidance_mode target"
python main.py --few_shot 64 --guidance_mode target
echo "finish --few_shot 64 --guidance_mode target"

echo "start --few_shot 64 --guidance_mode input"
#python main.py --few_shot 64 --guidance_mode input
echo "finish --few_shot 64 --guidance_mode input"


## 100-shot
echo "start --few_shot 100 --train_t5_tagger --if_spacy"
#python main.py --few_shot 100 --train_t5_tagger --if_spacy
echo "finish --few_shot 100 --train_t5_tagger --if_spacy"

echo "start --few_shot 100 --guidance_mode target"
python main.py --few_shot 100 --guidance_mode target
echo "finish --few_shot 100 --guidance_mode target"

echo "start --few_shot 100 --guidance_mode input"
#python main.py --few_shot 100 --guidance_mode input
echo "finish --few_shot 100 --guidance_mode input"


## 10-shot
echo "start --few_shot 10 --train_t5_tagger --if_spacy"
#python main.py --few_shot 10 --train_t5_tagger --if_spacy
echo "finish --few_shot 10 --train_t5_tagger --if_spacy"

echo "start --few_shot 10 --guidance_mode target"
python main.py --few_shot 10 --guidance_mode target
echo "finish --few_shot 10 --guidance_mode target"

echo "start --few_shot 10 --guidance_mode input"
#python main.py --few_shot 10 --guidance_mode input
echo "finish --few_shot 10 --guidance_mode input"