## 64-shot
echo "start --few_shot 64 --train_t5_tagger --if_spacy --use_t5_tagger"
python main.py --few_shot 64 --train_t5_tagger --if_spacy --use_t5_tagger
#python main.py --few_shot 64 --train_t5_tagger --use_t5_tagger
echo "finish --few_shot 64 --train_t5_tagger --if_spacy --use_t5_tagger"

echo "start --few_shot 64 --guidance_mode target --use_t5_tagger"
#python main.py --few_shot 64 --guidance_mode target --use_t5_tagger
echo "finish --few_shot 64 --guidance_mode target --use_t5_tagger"

echo "start --few_shot 64 --guidance_mode input --use_t5_tagger"
python main.py --few_shot 64 --guidance_mode input --use_t5_tagger
echo "finish --few_shot 64 --guidance_mode input --use_t5_tagger"


## 100-shot
echo "start --few_shot 100 --train_t5_tagger --if_spacy --use_t5_tagger"
python main.py --few_shot 100 --train_t5_tagger --if_spacy --use_t5_tagger
#python main.py --few_shot 100 --train_t5_tagger --use_t5_tagger
echo "finish --few_shot 100 --train_t5_tagger --if_spacy --use_t5_tagger"

echo "start --few_shot 100 --guidance_mode target --use_t5_tagger"
#python main.py --few_shot 100 --guidance_mode target --use_t5_tagger
echo "finish --few_shot 100 --guidance_mode target --use_t5_tagger"

echo "start --few_shot 100 --guidance_mode input --use_t5_tagger"
python main.py --few_shot 100 --guidance_mode input --use_t5_tagger
echo "finish --few_shot 100 --guidance_mode input --use_t5_tagger"


## 10-shot
echo "start --few_shot 10 --train_t5_tagger --if_spacy --use_t5_tagger"
python main.py --few_shot 10 --train_t5_tagger --if_spacy --use_t5_tagger
#python main.py --few_shot 10 --train_t5_tagger --use_t5_tagger
echo "finish --few_shot 10 --train_t5_tagger --if_spacy --use_t5_tagger"

echo "start --few_shot 10 --guidance_mode target --use_t5_tagger"
#python main.py --few_shot 10 --guidance_mode target --use_t5_tagger
echo "finish --few_shot 10 --guidance_mode target --use_t5_tagger"



echo "start --few_shot 10 --guidance_mode input --use_t5_tagger"
python main.py --few_shot 10 --guidance_mode input --use_t5_tagger
echo "finish --few_shot 10 --guidance_mode input --use_t5_tagger"