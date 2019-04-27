`safe-debates`
=====
_A PyTorch implementation of AI safety via debate_

This is currently not maintained and **very** unoptimized.  You can view the tracing files in `profiles/` to understand why -- it's essentially a poor choice of data structure (Python lookups are _awfully_ slow).  Also, there are quite a few TODOs throughout the code.

If anything, this repo should be viewed as a set of design choices and abstractions for developing agents with debate.  If you use it, please reference it.

Based on [`cle-mnist`](https://github.com/jvmancuso/cle-mnist).

### References
- [arxiv](https://arxiv.org/abs/1805.00899)
- [blog post](https://blog.openai.com/debate/)
- [debate game website](https://debate-game.openai.com/)

### Experimental Results
**Sparse classifier on random data**

Density | Accuracy | Average Cross-Entropy
--- | --- | ---
6px | 57.4% | 1.1948
4px | 46.7% | 1.4775

**Commands to reproduce**<br>
`python train_judge.py --pixels 6 --seed 4224 --checkpoint-filename 6px`<br>
`python train_judge.py --pixels 4 --batches 50000 --seed 4224 --checkpoint-filename 4px`
