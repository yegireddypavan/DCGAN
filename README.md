🧠 DCGAN — Deep Convolutional GAN for Multi-Dataset Image Generation
A Stability-Oriented GAN Implementation on CelebA, MNIST & SVHN

This repository presents a custom Deep Convolutional Generative Adversarial Network (DCGAN) designed to evaluate adversarial training stability across datasets of varying complexity — from handwritten grayscale digits to real-world face images.

🚀 Project Highlights

✔ Stability-focused GAN training
✔ Supports CelebA, MNIST, and SVHN datasets
✔ Binary cross-entropy loss for improved convergence
✔ Tools for diagnosing mode collapse and adversarial imbalance
✔ Suitable for machine learning research, demonstrations, and academic papers

📌 Key Research Goals

This project studies how DCGAN stability is affected by:

Dataset complexity differences

Generator vs. discriminator learning balance

Batch sizes and label smoothing

Latent-space structure & expressiveness

Training logs include:

Loss evolution (G-loss vs D-loss)

Discriminator accuracy trends

Gradient norm monitoring

Latent-space interpolation

Discriminator score distributions

🗂️ Supported Datasets
Dataset	Image Size	Characteristics	Storage Path
CelebA	64×64 RGB	Human faces	datasets/celeba/img_align_celeba/
MNIST	28×28 grayscale	Handwritten digits	Dataset/mnist/
SVHN	32×32 RGB	Street-view digits	Dataset/SVHN/

Note: CelebA must be downloaded manually due to license restrictions.

🛠️ Environment & Dependencies

Tested using:

Python ≥ 3.8

TensorFlow / Keras

NumPy

Matplotlib

tqdm

scikit-image

Install all dependencies:

```bash
pip install tensorflow numpy matplotlib tqdm scikit-image
```

🧱 Model Architecture
🔹 Generator

Dense projection + reshape

Series of Conv2DTranspose (upsampling) blocks

BatchNorm + LeakyReLU

Tanh output (normalized image limits)

🔸 Discriminator

Conv2D downsampling layers

LeakyReLU activations

Optional dropout

Sigmoid final prediction

💡 Binary cross-entropy → more stable than MSE used in LSGAN

🏋️ Training Strategy
1️⃣ Discriminator Update

Real images → label 1
Fake images → label 0
Supports label smoothing for stability

2️⃣ Generator Update

Goal → fool the discriminator into predicting real labels

📊 Per-epoch logging:

D-Loss & G-Loss

Accuracy for real/fake samples

Generated image sampling

All plots and evaluation results are automatically saved.

📈 Outputs & Analysis Tools

Generated diagnostics include:

Tool	Purpose
Loss curves	Indicator of convergence stability
Discriminator accuracy curve	Detecting imbalance (ideal: 45–55%)
Latent-space interpolation	Continuity of learned features
D-score histogram	Detects overconfidence
Gradient-norm plots	Spotting collapse or exploding gradients
Evaluation tables	Final quantitative comparisons
📚 Comparison With LSGAN (Research Paper)
Dataset	LSGAN Behavior	This Implementation
MNIST	Stable	Smoother convergence
SVHN	Often unstable	Stable using BCE
CelebA	Mode collapse common	Avoids collapse + balanced accuracy

🏆 Improvements due to:

Binary cross-entropy

BatchNorm + LeakyReLU

Label smoothing

Gradient norm supervision

▶️ Run Training

CelebA:
```bash
python dcgan_stability_analysis.py --dataset celebA
```

MNIST:
```bash
python dcgan_stability_analysis.py --dataset mnist
```

SVHN:
```bash
python dcgan_stability_analysis.py --dataset svhn
```

Additional hyper-parameters (batch size, epochs, latent dimension) can be modified via command-line flags.

❗ Troubleshooting
Issue	Cause	Fix
Black generated images	Generator collapse	Reduce LR / increase batch
D-accuracy ≈ 1.0	Discriminator overpowering	Apply label smoothing / lower D-LR
Slow CelebA training	Large dataset	Reduce resolution or dataset size
Empty plots	Logging disabled	Move logging inside epoch loop
🙏 Acknowledgements

Datasets:

CelebA — Chinese University of Hong Kong

SVHN — Stanford UFLDL Lab

MNIST — Yann LeCun et al.

References:

DCGAN (Radford et al.)

LSGAN research paper (baseline comparison)
