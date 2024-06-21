root = "/data/mathieu/" # local path to the the directory where PromptSum is cloned
cache_path = '/data/mathieu/hf_models/pegasus-large/' # local path to cache directory with the backbone's model weights
pretrain_ckpt = "pretrained_ckpt/019/bestckpt_full_model" # local path to PromptSum pretrained backbone
pretrain_prompt_ckpt = "pretrained_ckpt/019/bestckpt_prompt" # local path to PromptSum pretrained soft prompts

# all the datasets considered in our experiments
dataset_names = ["ccdv/cnn_dailymail", "xsum", "reddit_tifu", "wikihow", "billsum", "samsum"]

dataset_versions = ["3.0.0", "default", "long", "all", "default", "samsum"]
text_keys = ["article", "document", "documents", "text", "text", "dialogue"]
summary_keys = ["highlights", "summary", "tldr", "headline", "summary", "summary"]
validation_keys = ["validation", "validation", "", "validation", "test", "validation"]
test_keys = ["test", "test", "", "test", "test", "test"]
highlights = [True, False, False, False, False, False]
max_lengths = [512, 512, 512, 512, 1024, 512]
max_position_embeddings = [1024, 1024, 1024, 1024, 1536, 1024]
max_summary_lengths = [128, 64, 64, 128, 256, 64]
val_sizes = [13368, 11332, 4213, 5600, int(0.1 * 18949), 818]
test_sizes = [11490, 11334, 4222, 5600, 3269, 819]