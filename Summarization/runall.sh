## 10-shot
## train t5 tagger
echo "start --few_shot 10 --train_t5_tagger"
python main.py --few_shot 10 --train_t5_tagger
echo "finish --few_shot 10 --train_t5_tagger"
## oracle: target (T5 entity)
echo "start --few_shot 10 --guidance_mode oracle"
python main.py --few_shot 10 --guidance_mode oracle
echo "finish --few_shot 10 --guidance_mode oracle"
## T5 tagger (generate entities in summary)
echo "start --few_shot 10 --guidance_mode nomral"
python main.py --few_shot 10 --guidance_mode nomral
echo "finish --few_shot 10 --guidance_mode nomral"

## 64-shot
## train t5 tagger
echo "start --few_shot 64 --train_t5_tagger"
python main.py --few_shot 64 --train_t5_tagger
echo "finish --few_shot 64 --train_t5_tagger"
## oracle: target (T5 entity)
echo "start --few_shot 64 --guidance_mode oracle"
python main.py --few_shot 64 --guidance_mode oracle
echo "finish --few_shot 64 --guidance_mode oracle"
## T5 tagger (generate entities in summary)
echo "start --few_shot 64 --guidance_mode nomral"
python main.py --few_shot 64 --guidance_mode nomral
echo "finish --few_shot 64 --guidance_mode nomral"


## 100-shot
## train t5 tagger
echo "start --few_shot 100 --train_t5_tagger"
python main.py --few_shot 100 --train_t5_tagger
echo "finish --few_shot 100 --train_t5_tagger"
## oracle: target (T5 entity)
echo "start --few_shot 100 --guidance_mode oracle"
python main.py --few_shot 100 --guidance_mode oracle
echo "finish --few_shot 100 --guidance_mode oracle"
## T5 tagger (generate entities in summary)
echo "start --few_shot 100 --guidance_mode nomral"
python main.py --few_shot 100 --guidance_mode nomral
echo "finish --few_shot 100 --guidance_mode nomral"