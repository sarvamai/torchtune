from transformers import AutoTokenizer, LlamaTokenizer
from torchtune.models.sarvam1 import sarvam1_tokenizer
from tqdm import tqdm

hf_llama = LlamaTokenizer(vocab_file='/home/rahul_sarvam_ai/nemo_models/original_tokenizer/updated_tokenizer.model', legacy=False)
tt_tokenizer = sarvam1_tokenizer('/home/rahul_sarvam_ai/nemo_models/original_tokenizer/tokenizer.model')

hf_llama.chat_template = "{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'] %}{% else %}{% set loop_messages = messages %}{% set system_message = false %}{% endif %}\n{% for message in loop_messages %}\n{% if message['role'] not in ['user', 'assistant', 'tool_calls'] %}\n{{ raise_exception('Invalid role: ' + message['role'] + '. Must be user, assistant, or tool_calls.') }}\n{% endif %}\n{% if loop.index0 == 0 and system_message != false %}\n{% set content = '<<SYS>>\n' + system_message + '\n<</SYS>>\n\n' + message['content'] %}\n{% else %}\n{% set content = message['content'] %}\n{% endif %}\n{% if message['role'] == 'user' %}\n{{ bos_token + '[INST] ' + content.strip() + ' [/INST]' }}\n{% elif message['role'] == 'assistant' %}\n{{ ' ' + content.strip() + ' ' + eos_token }}\n{% elif message['role'] == 'tool_calls' %}\n{{ ' [TOOL_CALLS] ' + content.strip() + ' [/TOOL_CALLS] ' }}\n{% endif %}\n{% endfor %}"

import datasets
indic_ds = datasets.load_dataset('sarvam/indic-sft-dataset-sample-sep2024', split='train')
en_ds = datasets.load_dataset('sarvam/claude-generated-sft', 'indic-culture', split='train')
chat_ds = datasets.load_dataset('sarvam/norobots_sft', split='train')

indic_ds = indic_ds.map(lambda x: {'text': x['query'] + '\n' + x['response']}, remove_columns=indic_ds.column_names)
en_ds = en_ds.map(lambda x: {'text': x['question'] + '\n' + x['answer']}, remove_columns=en_ds.column_names)
chat_ds = chat_ds.map(lambda x: {'text': hf_llama.apply_chat_template([x['messages']], tokenize=False)[0]}, remove_columns=chat_ds.column_names)

ds = datasets.concatenate_datasets([indic_ds, en_ds])
for d in tqdm(ds):
    assert tt_tokenizer.encode(d['text'], add_bos=False, add_eos=False) == hf_llama.encode(d['text'], add_special_tokens=False), f"Mismatch for {d['text']}" 

for d in tqdm(chat_ds):
    assert tt_tokenizer.encode(d['text'], add_bos=False, add_eos=False) == hf_llama.encode(d['text'], add_special_tokens=False), f"Mismatch for {d['text']}" 
