## 10-shot
echo "start 10-shot finetune_entity"
python main.py --few_shot 10 --finetune_entity
echo "end 10-shot finetune_entity"

echo "start 10-shot finetune_summary"
python main.py --few_shot 10 --finetune_summary --infer_val_entities
echo "end 10-shot finetune_summary"

echo "start 10-shot finetune_summary oracle"
python main.py --few_shot 10 --finetune_summary --infer_val_entities --guidance_mode target
echo "end 10-shot finetune_summary oracle"



## 64-shot
echo "start 64-shot finetune_entity"
python main.py --few_shot 64 --finetune_entity
echo "end 64-shot finetune_entity"

echo "start 64-shot finetune_summary"
python main.py --few_shot 64 --finetune_summary --infer_val_entities
echo "end 64-shot finetune_summary"

echo "start 64-shot finetune_summary oracle"
python main.py --few_shot 64 --finetune_summary --infer_val_entities --guidance_mode target
echo "end 64-shot finetune_summary oracle"



## 100-shot
echo "start 100-shot finetune_entity"
python main.py --few_shot 100 --finetune_entity
echo "end 100-shot finetune_entity"

echo "start 100-shot finetune_summary"
python main.py --few_shot 100 --finetune_summary --infer_val_entities
echo "end 100-shot finetune_summary"

echo "start 100-shot finetune_summary oracle"
python main.py --few_shot 100 --finetune_summary --infer_val_entities --guidance_mode target
echo "end 100-shot finetune_summary oracle"