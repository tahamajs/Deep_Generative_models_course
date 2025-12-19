1. Large Language Models & Reasoning (LLMs)
   The focus has shifted from simple "next-token prediction" to internal reasoning and post-training scaling.

DeepSeek-R1 (Shao et al., 2024): Revolutionized open-source reasoning by using Reinforcement Learning (GRPO) to achieve O1-level performance.

The Llama 3 Herd of Models (Grattafiori et al., 2024): The blueprint for modern foundation models, scaling to 405B parameters with high-density data.

Kimi K1.5 (Kimi Team, 2025): A major 2025 paper on scaling Reinforcement Learning to overcome the limits of human-written training data.

Quiet-STaR (Zelikman et al., 2024): Proposes a way for models to generate "invisible thoughts" before every word they speak to improve accuracy.

Gemma 2 Technical Report (Team Gemma, 2024): Introduced "distillation-on-the-fly," allowing smaller models (9B/27B) to punch way above their weight class.

Phi-3 Technical Report (Abdin et al., 2024): Demonstrated that "data quality is all you need" by training a 3.8B model that rivals GPT-3.5 using textbook-style data.

Direct Language Model Alignment from Online AI Feedback (Guo et al., 2024): A move toward RLAIF, where AI evaluates AI to speed up safety and accuracy training.

BitNet b1.58 (Ma et al., 2024): Introduced the "1-bit LLM," where weights are just {−1,0,1}, promising massive energy savings for future hardware.

Mixture-of-Experts (MoE) in Qwen2 (Yang et al., 2024): Detailed how to manage massive expert models efficiently for coding and mathematics.

Chain-of-Thought Empowers LLMs to Solve Math (2025): A deep dive into how "system 2" thinking in models is achieved via iterative verification.

Buffet of Thoughts (2025): A leading paper on "Thought-Augmented Reasoning," which uses a dynamic buffer to store and reuse reasoning paths.

2. Vision, Video & Multimodal Models
   Generative vision is moving away from static pixels toward "world simulation" and real-time video understanding.

SAM 2: Segment Anything in Images and Videos (Ravi et al., 2025): Meta’s breakthrough in real-time video segmentation and tracking using a streaming memory bank.

Visual Autoregressive Modeling (VAR) (Tian et al., 2024): Winner of NeurIPS 2024. It treats image generation as a multi-scale sequence, making it faster and more scalable than Diffusion.

Sora: Video Generation as World Simulators (OpenAI, 2024): Though a report, it established the Diffusion Transformer (DiT) as the industry standard for consistent video.

Depth Anything (Yang et al., 2024): A massive leap in monocular depth estimation, using large-scale unlabeled data to create "foundation models" for 3D vision.

Vision Transformers Need Registers (Darcet et al., 2024): Identified and fixed "spiky" artifacts in ViTs, significantly improving object detection performance.

DeepSeek-VL (Lu et al., 2024): An open-source VLM that excels in real-world visual grounding and GUI navigation.

CogAgent (Hong et al., 2024): A 18B parameter model specialized in "seeing" and navigating computer screens (GUI agents).

4D Gaussian Splatting (Wu et al., 2024): Enabled real-time rendering of dynamic scenes, allowing AI to generate interactive 3D video.

V-JEPA (Meta, 2024): Yann LeCun’s paper on non-generative world models that learn by predicting high-level "latent" features rather than individual pixels.

Lumina-T2X (2024): A unified transformer architecture capable of generating images, videos, and music from a single set of weights.

3. Deep Generative Models & Efficiency
   Researchers are focused on making generation faster (1-step sampling) and moving beyond standard Transformers.

Mamba: Linear-Time Sequence Modeling (Gu & Dao, 2024): The paper that introduced Selective State-Space Models (SSMs) as a faster, linear-scaling alternative to Transformers.

Flow Matching for Generative Modeling (Lipman et al., 2024): The theoretical foundation for the newest generation of models (like Flux.1), which are easier to train than Diffusion.

Phased Consistency Models (PCM) (Luo et al., 2024): Allows high-quality diffusion generation in just 1-4 steps instead of the typical 50.

KAN: Kolmogorov-Arnold Networks (Liu et al., 2024): A complete rethinking of neural networks, replacing fixed weights with learnable functions on the edges of the graph.

Rectified Flow Transformers (Esser et al., 2024): The math behind Stable Diffusion 3, improving the "straightness" of the generation path for better prompt adherence.

Vision Mamba (Vim) (Zhu et al., 2024): Successfully applied the Mamba architecture to vision tasks, proving Transformers aren't the only option for high-res images.

Jamba (AI21, 2024): The first large-scale "Hybrid" model, mixing Transformer and Mamba layers to get the best of both worlds (memory + speed).

The AdEMAMix Optimizer (2025): A new optimization algorithm that outperforms Adam/AdamW by managing multi-scale gradients more effectively.

Sparse Attention by DeepSeek (2025): Detailed how to achieve near-infinite context windows (up to 1M tokens) while keeping the compute cost low.

Token Merging for Training-Free Binding (2024): A technique to make image generators follow complex prompts (e.g., "a red cat in a blue hat") without mixing up the colors.

4. Deep Learning for Science (AI4Science)
   This category represents the "real-world" impact of deep learning, particularly in biology and chemistry.

AlphaFold 3 (Abramson et al., 2024): Expanded structure prediction to almost all biological molecules, including DNA and RNA.

The AI Scientist (Lu et al., 2024): An end-to-end system where AI generates a scientific idea, runs experiments, writes the paper, and peer-reviews itself.

Equivariant Neural Diffusion for Molecule Generation (2024): A specialized generative model that respects the laws of physics to design new drug molecules.

Generative Modeling of Molecular Dynamics (2025): A paper using deep learning to simulate the movement of atoms millions of times faster than traditional physics simulations.

IgGM: Generative Model for Antibody Design (2025): A breakthrough in generative medicine for creating synthetic antibodies to fight diseases.

Deep Genomics (2025): A study on using deep learning to predict how specific genetic mutations will impact human health.

Graph Neural Networks for Weather Forecasting (GraphCast, 2024): Proved that deep learning on graphs can outperform the world's best supercomputer-based weather models.

Foundation Models for Material Science (GNoME, 2024): DeepMind's discovery of millions of new crystal structures using deep generative networks.

DAGER: Exact Gradient Inversion (2024): A critical paper on privacy, showing how sensitive data can be protected during large-scale scientific training.

Neural PDE Solvers (2025): Research into using deep learning to solve Partial Differential Equations, the language of physics, in real-time.
