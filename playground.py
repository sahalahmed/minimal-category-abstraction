from transformers import (
    CONFIG_MAPPING,
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AdamW,
    PreTrainedModel,
    PreTrainedTokenizer,
    TextDataset,
    get_constant_schedule,
    get_linear_schedule_with_warmup,
    AutoModelForPreTraining,
    BatchEncoding,
    default_data_collator,
)
import torch
import scipy

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

def switch_model_checkpoint(model, checkpoint_path, tokens):
    try:
        # if there are saved novel word embeddings, then simply load and set them
        novel_word_embeddings = torch.load(checkpoint_path + '/' + 'novel_word_embeddings.pt')

    except FileNotFoundError:
        # otherwise, assume there is a full checkpoint and load it
        model = model_class.from_pretrained(checkpoint_path)

    else:
        embs = novel_word_embeddings[0]
        embeddings_weight = get_embeddings_weight(model)
        with torch.no_grad():
            for token_id, embedding in zip(tokens, embs):
                embeddings_weight[token_id] = embedding
        print(embs)

    return model

def get_embeddings_weight(model):
    embeddings = model.resize_token_embeddings()
    return embeddings.weight

def embedding_distance(model, token1, token2):
    word_embeddings = model.base_model.embeddings.word_embeddings.weight
    embedding_distance = scipy.spatial.distance.cosine(word_embeddings[token1].detach().numpy(), word_embeddings[token2].detach().numpy())
    # print(tokenizer.decode(token1))
    # print(tokenizer.decode(token2))
    return embedding_distance
    
def closest_word(model, token1, token2):
    token1_closest = [0, 2]
    token2_closest = [0, 2]

    for i in range(999, 30522):
        if embedding_distance(model, token1, i) < token1_closest[1] and i != token1:
            token1_closest = [i, embedding_distance(model, token1, i)]
            print(f"New token found close to token 1. Token is {tokenizer.decode(i)} at step {i}.")
        if embedding_distance(model, token2, i) < token2_closest[1] and i != token2:
            token2_closest = [i, embedding_distance(model, token2, i)]
            print(f"New token found close to token 2. Token is {tokenizer.decode(i)} at step {i}.")

    return token1_closest, token2_closest

             
        

model_class = AutoModelForMaskedLM

model = model_class.from_pretrained(
            "bert-base-uncased",
            from_tf=False,
            config = AutoConfig.from_pretrained("bert-base-uncased", cache_dir=None, output_hidden_states=True),
            cache_dir=None,
        )

model = switch_model_checkpoint(model, "checkpoints/bert-base-uncased/unused_pairs_1/testexp_nv_unused_token_numbers_1_2_learning_rate_0.001_seed_1/checkpoint-69", [2,3])
# print(embedding_distance(model, 999, 3))
print(closest_word(model, 2, 3))
# print(tokenizer.encode("you"))



# model, token_ids(2) --> distance between the token_ids