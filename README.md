CURL Rainbow
=======
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

**Status**: Archive (code is provided as-is, no updates expected)

This is an implementation of [CURL: Contrastive Unsupervised Representations for
Reinforcement Learning](https://arxiv.org/abs/2004.04136) coupled with the [Data Efficient Rainbow method](https://arxiv.org/abs/1906.05243) for Atari
games. The code by default uses the 100k timesteps benchmark and has not been
tested for any other setting.

Run the following command (or `bash run_curl.sh`) with the game as an argument:

```
python3 main.py --game ms_pacman
```

To install all dependencies, run `bash install.sh`.
