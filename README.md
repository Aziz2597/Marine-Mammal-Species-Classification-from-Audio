# Marine Bioacoustics — Species Classification & Generative Augmentation

This project implements a comprehensive deep learning system for classifying marine mammal species from short audio recordings and for generating synthetic spectrograms and audio using VAE and conditional diffusion models. The code compares multiple model families (DNN, 2D-CNN, RNN, LSTM, Transformer, 1D-CNN on raw audio), includes training and evaluation utilities, and provides end-to-end functionality for:

- Preprocessing and feature extraction (mel-spectrograms)  
- Training and evaluating classifiers with proper splitting and checkpoints  
- Generating synthetic data with a VAE and a class-conditioned diffusion model  
- Reconstructing audio from generated spectrograms (Griffin-Lim)  
- Visualizing latent structure (UMAP / t-SNE) and generated vs real spectrograms  
- Saving audios and metrics to Google Drive for grading and inspection  

---

## Real-World Significance

Passive acoustic monitoring is widely used for marine biodiversity assessment and conservation. Manual audiogram annotation is slow and expertise-heavy. This project provides an automated, reproducible pipeline to:

- Classify marine mammal species from hydrophone recordings  
- Produce synthetic audio and spectrograms to augment training data  
- Provide interpretable latent visualizations and quantitative evaluation  
- Be readily reproducible in Google Colab and transferable to production inference pipelines  

---

## Dataset & Expected Layout

### Audio Files
WAV files (or other readable audio) kept in a folder. Default used by the notebook:
```bash
/content/drive/MyDrive/wmms_audio/
```

### Labels CSV
CSV with at least two columns: `file` (filename) and `species` (label). Default:
```bash
/content/drive/MyDrive/wmms_labels.csv
```

---

## Key Dataset Characteristics Used by the Code

- Samples: ~1,357 labelled clips across 32 species  
- Audio standardization: resampled to 22,050 Hz  
- Clip length: standardized to ~3 seconds → 66,150 samples  
- Input features: 96-bin mel-spectrograms sized 96 × 112 (n_mels × time frames)  
- FFT / hop: `n_fft = 2048`, `hop_length = 512`  
- Train/Validation/Test split: 70 / 15 / 15 (stratified by class where applicable)  

---

## High-Level Architecture & Design

### Feature Pipelines

- **Spectrogram pipeline:** `librosa` → mel-spectrogram → dB power → pad/truncate to fixed width  
- **Raw audio pipeline:** 1D waveform padded/truncated to fixed length for 1D-CNN experiments  

### Model Families Implemented

- **DNN:** Flattened spectrogram baseline with batch normalization and dropout  
- **2D-CNN:** Multi-layer convolutional extractor with adaptive pooling and linear classifier  
- **RNN / LSTM:** Sequence models over time frames (spectrogram time axis)  
- **Transformer:** Positional encodings + transformer encoder for long-range dependencies  
- **RawAudio 1D-CNN:** Temporal convolutions directly on waveform  
- **Contrastive (SupCon):** Contrastive pretraining + classifier fine-tuning  
- **VAE:** LightVAE — convolutional encoder/decoder (latent dim default = 64) for spectrogram generation and augmentation  
- **Conditional Diffusion:** Class-conditioned DDPM-style architecture for spectrogram generation  

---

## Evaluation & Visualizations

### Metrics Computed / Available

- Multi-class accuracy  
- Confusion matrix  
- Multi-class ROC (One-vs-Rest) with AUC using stored probabilities  
- Reconstruction metrics (e.g., spectrogram MSE, waveform SNR) for VAE samples  

### Visualizations Generated

- Training and validation loss & accuracy curves  
- Per-class ROC curves with AUC legends  
- Confusion matrix with class labels  
- VAE: original vs generated spectrogram grid  
- VAE latent space: UMAP and t-SNE projections  
- Diffusion: generated spectrogram sample grids  

---

## Reproducibility & How to Run (Colab Recommended)

- **Runtime:** Use Google Colab with a GPU runtime (T4 or better recommended). Ensure runtime type is set to GPU.  
- **Mount Google Drive:** Update the notebook paths to point to your Drive locations for audio and CSV files.

---

## Limitations & Important Caveats

- **Dataset size & imbalance:** Results are sensitive to class imbalance and small sample counts per species. Augmentation helps but must be validated carefully.  
- **Audio inversion fidelity:** Griffin-Lim is a basic inversion method. For higher quality, consider neural vocoders (MelGAN, WaveGlow) or training models on linear-power spectrograms.  
- **Compute requirements:** Diffusion training is compute-intensive. Adjust timesteps and sampling steps or pre-generate synthetic data when needed.  
- **Metric validity:** AUC/ROC per class may be noisy for very small classes — interpret such metrics cautiously.  

---

## Future Enhancements

- Replace Griffin-Lim with a neural vocoder for improved audio fidelity  
- Use transfer learning with pretrained audio backbones  
- Explore stronger data augmentation (time-stretch, pitch shift, additive noise)  
- Implement more robust conditional sampling for the diffusion model (classifier guidance)  
- Produce an ensemble across top-performing architectures for improved classification  
