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

`PackedPretrainDataset` in `trainer/train_pretrain.py` loads JSON/JSONL records with a `text` field through `load_dataset`. A positive `subset` keeps the first N records; the training CLI defaults to `-1` to use all records.

- **On-demand packing:** `__getitem__(index)` starts with `samples[index]`, tokenizes it, adds BOS/EOS, and truncates to `max_pos`. Random samples fill the remaining space until the row contains `max_length + 1` tokens. Tokens and packs are not cached.
- **Coverage:** Each source sample starts one row per completed shuffled epoch; random fillers may repeat. Overflow is discarded, so long samples' tails are not guaranteed to be used.
- **Labels:** `_make_pack` shifts inputs and labels by one for next-token prediction, producing length `max_length` (default 512). BOS targets and padding labels are `-100`, ignored by the loss.
- **Attention:** Positions restart at zero for each sample occurrence. A causal mask restricts attention to the same occurrence. Padding keys are masked, with self-attention allowed for padding queries; normally packing fills the row completely.
- **Outputs:** Returns `(input_ids, label_ids, position_idxs, attention_mask, non_pad_mask)`, with shapes `[L]`, `[L]`, `[L]`, `[L, L]`, and `[L]`. Boolean masks use `True` for allowed attention or valid tokens.
- **DataLoader:** Defaults to batch size 32, shuffling, and 8 workers, with pinned memory on CUDA and persistent workers when enabled. Temporary tokenization and prefetched batches still use RAM.
