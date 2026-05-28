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
* weights of the model available on huggings faces (too big for github), check below for the link. 

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### ⚙️ Installation & Requirements

All the necessary dependencies to run, train, and test the **Zero-Ping** codec are listed in the `requirements.txt` file. 

It is highly recommended to set up a virtual environment (using `venv` or `conda`) before installing the packages to avoid version conflicts. Once your environment is active, you can install everything by running the following command in your terminal:

```bash
pip install -r requirements.txt
```

The pre-trained version of **Zero-Ping** is publicly available on Hugging Face:

[https://huggingface.co/Lucabr01/Zero-Ping](https://huggingface.co/Lucabr01/Zero-Ping)

The Hugging Face repository contains the model checkpoint and the instructions required to use it for inference. Check it out!

### Try the Model

An inference test for **Zero-Ping** is available through the following Google Colab notebook:

[https://colab.research.google.com/drive/1S5OawwYIjd88uhVp1Vs7dog4gKcdvo-y?usp=sharing](https://colab.research.google.com/drive/1S5OawwYIjd88uhVp1Vs7dog4gKcdvo-y?usp=sharing)

The notebook allows users to run the pre-trained model directly and test the codec on custom audio files. A GPU runtime is not required for standard inference, since the model is lightweight and runs quickly on CPU. Using a GPU may only be useful when processing very long audio files.

