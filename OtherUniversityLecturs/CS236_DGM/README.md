
![Hits](https://hitcounter.pythonanywhere.com/count/tag.svg?url=https%3A%2F%2Fgithub.com%2FSKKSaikia%2FCS236_DGM)

# CS236 : Deep Generative Models

<img src="https://github.com/SKKSaikia/CS236_DGM/blob/master/cs236_c.jpg">

## Table of Contents
1. [Course Overview](#course-overview)
2. [Repository Structure](#repository-structure)
3. [Homeworks](#homeworks)
4. [How to Run & Submit](#how-to-run--submit)
5. [Notes & Resources](#notes--resources)
6. [Final Project](#final-project)

---

## Course Overview

Generative models are widely used in many subfields of AI and Machine Learning. Recent advances in parameterizing these models using deep neural networks, combined with progress in stochastic optimization methods, have enabled scalable modeling of complex, high-dimensional data including images, text, and speech. In this course, we study the probabilistic foundations and learning algorithms for deep generative models, including Variational Autoencoders (VAE), Generative Adversarial Networks (GAN), autoregressive models, and normalizing flow models. The course also discusses application areas that have benefitted from deep generative models, including computer vision, speech and natural language processing, graph mining, and reinforcement learning.

**Grading:** Homeworks (15% x 3 = 45%) + Midterm: 15% + Course Project 40%

---

## Repository Structure

```
HW1_Basic_VAE/
   src/              # main.py, model.py, dataset.py
   data/             # papers.csv
   checkpoints/      # checkpoint.pth
   configs/          # config.yml
   requirements/     # requirements.txt
   scripts/          # make_submission.sh
   docs/             # README
   report/           # nips_2018.sty, nips_2018.tex

HW2_Advanced_VAEs/
   src/              # run_vae.py, run_fsvae.py, run_gmvae.py, run_ssvae.py, codebase/
   requirements/     # requirements.txt
   scripts/          # make_submission.sh
   docs/             # README.md
   report/           # nips_2018.sty, nips_2018.tex

HW3_GANs/
   src/              # run_gan.py, run_conditional_gan.py, codebase/
   scripts/          # make_submission.sh
   report/           # nips_2018.sty, nips_2018.tex

Other folders: notes/, exam/, doc/, etc.
```

---

## Homeworks

### HW1: Basic VAE
- [PDF](hw/CS236_Homework_1.pdf)
- [Starter Code](hw/hw1/)
- [Solution](hw/CS236_hw1_answers.pdf)

### HW2: Advanced VAEs
- [PDF](hw/hw2.pdf)
- [Starter Code](hw/hw2-released/)
- [Solution](hw/CS236_hw2_answers.pdf)

### HW3: GANs
- [PDF](hw/CS236_Homework_3.pdf)
- [Starter Code](hw/hw3starter/)
- [Solution](hw/CS236_Homework_3_answer.pdf)

---


## How to Run & Submit

### HW1: Basic VAE
**Requirements:**
```
matplotlib
numpy
pyyaml
torch
```
Install:
```bash
pip install --user -r HW1_Basic_VAE/requirements/requirements.txt
```
Run:
```bash
python3 HW1_Basic_VAE/src/main.py
```
Submission:
```bash
bash HW1_Basic_VAE/scripts/make_submission.sh
# This will zip answers.pkl, samples.txt, shakespeare.png, random.png, nips.png, shakespeare_raw.pkl, random_raw.pkl, nips_raw.pkl, main.py, model.py
```
See HW1_Basic_VAE/docs/README for more details.

### HW2: Advanced VAEs
**Requirements:**
```
tqdm==4.20.0
numpy==1.15.2
torchvision==0.2.1
torch==0.4.1.post2
```
Install:
```bash
pip install --user -r HW2_Advanced_VAEs/requirements/requirements.txt
```
Run any model:
```bash
python3 HW2_Advanced_VAEs/src/run_vae.py
python3 HW2_Advanced_VAEs/src/run_fsvae.py
python3 HW2_Advanced_VAEs/src/run_gmvae.py
python3 HW2_Advanced_VAEs/src/run_ssvae.py
```
Submission:
```bash
bash HW2_Advanced_VAEs/scripts/make_submission.sh
# This will zip codebase/utils.py, codebase/models/vae.py, codebase/models/gmvae.py, codebase/models/ssvae.py, codebase/models/fsvae.py
```
See HW2_Advanced_VAEs/docs/README.md for full assignment details, file modification checklist, and tips for running on CPU.

### HW3: GANs
**Requirements:**
Use HW2 requirements (torch, numpy, etc.)
Run:
```bash
python3 HW3_GANs/src/run_gan.py
python3 HW3_GANs/src/run_conditional_gan.py
```
Submission:
```bash
bash HW3_GANs/scripts/make_submission.sh
# This will zip codebase/gan.py and out*/fake_0900.png
# Prints: "Submission created in hw3.zip"
```

---

## More Details & Tips

### HW1
- To run, use `python3 main.py` in the src folder.
- To install requirements, use `pip3 install --user -r requirements.txt`.
- Submission script zips all required outputs and code files.

### HW2
- Only modify: `codebase/utils.py`, `codebase/models/vae.py`, `codebase/models/gmvae.py`, `codebase/models/ssvae.py`, `codebase/models/fsvae.py` (bonus).
- Do not change hyperparameters unless starting from scratch.
- Models can take a while to run on CPU (see README.md for timing estimates).
- Useful functions: `codebase.utils.load_model_by_name`, sampling in model files, `numpy.swapaxes`, `torch.permute`, `matplotlib.pyplot.imshow`.
- Checklist of functions to implement (see HW2 README.md for order).

### HW3
- Submission script zips GAN code and generated images.
- See HW2 requirements for dependencies.

---

---

## Notes & Resources

- [Course Notes](https://deepgenerativemodels.github.io/notes/index.html)
- [Deep Learning Book](https://www.deeplearningbook.org/) ([PDF](doc/Deep%20Learning%20Book%20-%20Ian%20Goodfellow.pdf))
- [Stanford CS236 Official Site](https://deepgenerativemodels.github.io/)
- [PyTorch Introduction](notes/IntroductiontoPyTorch.pdf)
- [Generative Models from OpenAI](https://blog.openai.com/generative-models/)
- [Collection of generative models (wiseodd)](https://github.com/wiseodd/generative-models)

---

## Final Project

Project resources:
- [Project Guidelines](doc/CS236PosterGuidelines.pdf)
- [Project Proposal Guidelines](doc/CS236ProjectProposalGuidelines.pdf)
- [Final Report Guidelines](doc/CS236ProjectFinalReportGuide.pdf)
- [Project Examples](doc/CS236ProjectExamples.pdf)
- [Nips LaTeX Format](nips_style_files.zip)

---

For any questions or issues, please refer to the course notes or reach out via the course forum.
