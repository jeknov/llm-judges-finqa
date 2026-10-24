"""TEST-ONLY toy model: a byte-level BPE tokenizer and a 2-layer Llama with random weights, built locally
and deterministically, so the tiny profile and backend tests need no download. It stands in for the
owner's ~0.5B tiny-profile model until that choice is made (M4)."""

from __future__ import annotations

from pathlib import Path

from filelock import FileLock

CHAT_TEMPLATE = (
    "{% for m in messages %}<|{{ m['role'] }}|>\n{{ m['content'] }}<|end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>\n{% endif %}"
)
SPECIAL_TOKENS = ["<|pad|>", "<|end|>", "<|user|>", "<|assistant|>", "<|system|>"]
TOY_SEED = 20261022  # global seed; the toy model is test-only and never used for results
_CORPUS = (
    "Context: Table: Question: Stated answer: Answer: Give only the final answer, with units if any. "
    "Reason briefly, then give the final answer on its own last line as: Answer: <value> "
    "the revenue net income increase decrease total ratio percent million thousand billion change "
    "from to in of and for was is are year 2019 2018 2017 2016 2015 0 1 2 3 4 5 6 7 8 9 . , % $ ( ) | - "
)


def build_toy_model(directory: Path) -> Path:
    directory = Path(directory)
    with FileLock(str(directory) + ".lock"):
        if (directory / "config.json").exists():
            return directory
        import torch
        from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
        from transformers import LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast

        tok = Tokenizer(models.BPE())
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=600,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tok.train_from_iterator([_CORPUS] * 50, trainer=trainer)
        hf_tok = PreTrainedTokenizerFast(
            tokenizer_object=tok, pad_token="<|pad|>", eos_token="<|end|>", model_max_length=16384
        )
        hf_tok.chat_template = CHAT_TEMPLATE
        torch.manual_seed(TOY_SEED)
        config = LlamaConfig(
            vocab_size=len(hf_tok),
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=4,
            max_position_embeddings=16384,
            pad_token_id=hf_tok.pad_token_id,
            eos_token_id=hf_tok.eos_token_id,
            bos_token_id=None,
            tie_word_embeddings=False,
        )
        model = LlamaForCausalLM(config)
        directory.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(directory, safe_serialization=True)
        hf_tok.save_pretrained(directory)
    return directory
