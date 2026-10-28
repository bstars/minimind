# Minimind

This is a project to reimplement [minimind](https://github.com/jingyaogong/minimind) for learning purpose. 
I might modify some code structure in my personal style.

I might ask you questions about
- Code structure
- Code functionality
- Code API
- Algorithm
- Network architecture
- etc

You need to answer my question.

# Pretraining

### 1. Dataset

`PackedPretrainDataset` tokenizes each JSONL `text` without automatic special tokens, wraps it in BOS/EOS, and concatenates samples into full `max_length` rows. Samples may span rows; the final incomplete row is dropped, so emitted rows have no padding.

- `position_idxs` resets for each sample. The causal `attention_mask` isolates samples; `True` means attention is allowed.
- `labels` copy `input_ids`, with BOS set to `-100`. The loss shifts labels internally; the first token of each row is not a target.
- Each item returns `input_ids`, `labels`, `position_idxs`, `attention_mask`, and `non_pad_mask`. The model uses the position indices for RoPE, applies the attention mask, and uses the non-pad mask for auxiliary losses.
