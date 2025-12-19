#!/usr/bin/env python3
import os
import requests
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Paper mappings: title -> arxiv_id
PAPERS = {
    # LLM Papers
    "DeepSeek-R1": "2501.12948",
    "Llama_3_Herd_of_Models": "2407.21783",
    "Kimi_K1.5": "2501.03054",
    "Quiet-STaR": "2403.09629",
    "Gemma_2_Technical_Report": "2408.00118",
    "Phi-3_Technical_Report": "2404.14219",
    "Direct_Language_Model_Alignment_from_Online_AI_Feedback": "2402.04792",
    "BitNet_b1.58": "2402.17764",
    "Mixture-of-Experts_(MoE)_in_Qwen2": "2410.02760",
    "Chain-of-Thought_Empowers_LLMs_to_Solve_Math": "2501.12268",
    "Buffet_of_Thoughts": "2406.08479",

    # Vision Papers
    "SAM_2_Segment_Anything_in_Images_and_Videos": "2408.00714",
    "Visual_Autoregressive_Modeling_(VAR)": "2406.05245",
    "Sora_Video_Generation_as_World_Simulators": "2412.00752",
    "Depth_Anything": "2406.09414",
    "Vision_Transformers_Need_Registers": "2309.16588",
    "DeepSeek-VL": "2403.05425",
    "CogAgent": "2312.08914",
    "4D_Gaussian_Splatting": "2312.00735",
    "V-JEPA": "2409.01896",
    "Lumina-T2X": "2406.07125",

    # Generative Models & Efficiency
    "Mamba_Linear-Time_Sequence_Modeling": "2312.00752",
    "Flow_Matching_for_Generative_Modeling": "2210.02747",
    "Phased_Consistency_Models_(PCM)": "2403.03206",
    "KAN_Kolmogorov-Arnold_Networks": "2404.19756",
    "Rectified_Flow_Transformers": "2410.14677",
    "Vision_Mamba_(Vim)": "2408.06280",
    "Jamba": "2403.19887",
    "The_AdEMAMix_Optimizer": "2409.07440",
    "Sparse_Attention_by_DeepSeek": "2407.10969",
    "Token_Merging_for_Training-Free_Binding": "2406.12057",

    # AI4Science
    "AlphaFold_3": "2401.13859",
    "The_AI_Scientist": "2408.06292",
    "Equivariant_Neural_Diffusion_for_Molecule_Generation": "2407.10775",
    "Generative_Modeling_of_Molecular_Dynamics": "2407.08634",
    "IgGM_Generative_Model_for_Antibody_Design": "2407.08525",
    "Deep_Genomics": "2406.02680",
    "Graph_Neural_Networks_for_Weather_Forecasting": "2306.06079",
    "Foundation_Models_for_Material_Science_(GNoME)": "2407.10775",
    "DAGER_Exact_Gradient_Inversion": "2406.07284",
    "Neural_PDE_Solvers": "2406.14585"
}

FOLDERS = {
    "LLM": ["DeepSeek-R1", "Llama_3_Herd_of_Models", "Kimi_K1.5", "Quiet-STaR", "Gemma_2_Technical_Report", "Phi-3_Technical_Report", "Direct_Language_Model_Alignment_from_Online_AI_Feedback", "BitNet_b1.58", "Mixture-of-Experts_(MoE)_in_Qwen2", "Chain-of-Thought_Empowers_LLMs_to_Solve_Math", "Buffet_of_Thoughts"],
    "Vision": ["SAM_2_Segment_Anything_in_Images_and_Videos", "Visual_Autoregressive_Modeling_(VAR)", "Sora_Video_Generation_as_World_Simulators", "Depth_Anything", "Vision_Transformers_Need_Registers", "DeepSeek-VL", "CogAgent", "4D_Gaussian_Splatting", "V-JEPA", "Lumina-T2X"],
    "Generative": ["Mamba_Linear-Time_Sequence_Modeling", "Flow_Matching_for_Generative_Modeling", "Phased_Consistency_Models_(PCM)", "KAN_Kolmogorov-Arnold_Networks", "Rectified_Flow_Transformers", "Vision_Mamba_(Vim)", "Jamba", "The_AdEMAMix_Optimizer", "Sparse_Attention_by_DeepSeek", "Token_Merging_for_Training-Free_Binding"],
    "AI4Science": ["AlphaFold_3", "The_AI_Scientist", "Equivariant_Neural_Diffusion_for_Molecule_Generation", "Generative_Modeling_of_Molecular_Dynamics", "IgGM_Generative_Model_for_Antibody_Design", "Deep_Genomics", "Graph_Neural_Networks_for_Weather_Forecasting", "Foundation_Models_for_Material_Science_(GNoME)", "DAGER_Exact_Gradient_Inversion", "Neural_PDE_Solvers"]
}

def download_paper(paper_name, arxiv_id, folder):
    """Download a single paper from arXiv"""
    base_path = "/Users/tahamajs/Documents/uni/DGM/PaperSLecturs"
    folder_path = f"{base_path}/{folder}_Papers"

    # Ensure folder exists
    os.makedirs(folder_path, exist_ok=True)

    url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    filename = f"{paper_name}.pdf"
    filepath = os.path.join(folder_path, filename)

    try:
        print(f"Downloading {paper_name}...")
        response = requests.get(url, stream=True, timeout=30)
        response.raise_for_status()

        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"✓ Downloaded {paper_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {paper_name}: {e}")
        return False

def main():
    """Download all papers"""
    print("Starting paper downloads...")

    total_papers = 0
    successful_downloads = 0

    # Download papers using thread pool for efficiency
    with ThreadPoolExecutor(max_workers=3) as executor:
        futures = []

        for folder, papers in FOLDERS.items():
            for paper in papers:
                if paper in PAPERS:
                    future = executor.submit(download_paper, paper, PAPERS[paper], folder)
                    futures.append(future)
                    total_papers += 1

        # Wait for all downloads to complete
        for future in as_completed(futures):
            if future.result():
                successful_downloads += 1

    print(f"\nDownload complete! {successful_downloads}/{total_papers} papers downloaded successfully.")

if __name__ == "__main__":
    main()
