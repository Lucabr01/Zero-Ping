# 🛡️🎧 Zero-Ping

Who has not experienced the frustration of packet loss while playing online games? This project addresses packet loss concealment for real-time mono 16 kHz speech transmission in a "neural way". ___Zero-Ping___ is an end-to-end trained neural audio codec based on an RVQ-GAN architecture, designed to reconstruct the audio waveform even when some transmitted packets are lost.

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### 🗺️ Code Navigation

* **`GilbertElliot.py`** — Simulates packet loss over a network channel using a two-state Markov model (Good/Bad). Used during training to generate realistic burst-loss patterns.
* **`components.py`** — Causal convolutional encoder and decoder that form the backbone of the codec.
* **`repair.py`** — Transformer that reconstructs lost audio frames in the latent space after packet loss.
* **`model.py`** — Full codec model assembling all components (encoder, quantizer, repair, decoder) into a single module.
* **`losses.py`** — All generator-side loss functions used to train the codec toward high-quality speech reconstruction.
* **`msstftd_16k_speech.py`** — GAN discriminator and its associated loss functions, adapted for 16 kHz mono speech.
* **`trainer.py`** — Training wrapper that coordinates the codec, discriminator, and losses in the correct D-step / G-step order.
* **`Utils.py`** — Low-level building blocks shared across the codebase (custom layers, activations).

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

