for type in None,'PERSON','NORP','FAC','ORG','GPE','LOC','PRODUCT','EVENT','WORK_OF_ART','LAW','LANGUAGE','DATE','TIME','PERCENT','MONEY','QUANTITY','ORDINAL','CARDINAL'
do
    echo "python main.py --few_shot=64 --filter_type=$type --guidance_mode=target_unique_filtered --pretrain=False"
    python main.py --few_shot=64 --filter_type=$type --guidance_mode=target_unique_filtered --pretrain=False > log/$type.txt
    echo "python main.py --few_shot=64 --filter_type=$type --guidance_mode=target_unique_filtered --pretrain=False"
done