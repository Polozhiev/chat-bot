class Params():
    max_new_tokens: int = 40
    repetition_penalty: float = 1.5
    do_sample: bool = True
    top_k: int = 10
    top_p: float = 0.95
    temperature: float = 1.2
    num_beams: int = 3
    no_repeat_ngram_size: int = 3
    length_penalty: float = 1.5
    num_return_sequences: int = 3
    