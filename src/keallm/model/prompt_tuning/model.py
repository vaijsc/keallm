from peft import  get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit, LoraConfig
from transformers import AutoModelForCausalLM


def get_pt_model(model_args, finetuning_args, model: "AutoModelForCausalLM"):
    tuning_config = PromptTuningConfig(
        task_type=TaskType.CAUSAL_LM, #This type indicates the model will generate text.
        prompt_tuning_init=PromptTuningInit.RANDOM,  #The added virtual tokens are initializad with random numbers
        num_virtual_tokens=model_args.num_query_tokens, #Number of virtual tokens to be added and trained.
    )

    peft_model = get_peft_model(model, tuning_config)
    return peft_model

def get_lora_model(model_args, finetuning_args, model: "AutoModelForCausalLM"):
    tuning_config = LoraConfig(
        r=finetuning_args.lora_rank,
        lora_alpha=finetuning_args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=finetuning_args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    peft_model = get_peft_model(model, tuning_config)
    return peft_model