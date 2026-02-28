"""
Generate a synthetic dataset of 3000+ AI/ML research papers with realistic distributions.

The dataset simulates metadata from AI/ML research papers spanning 2018-2025,
capturing trends like the rise of transformer architectures, increasing paper
volume, and power-law citation distributions.
"""

import csv
import random
import math
import hashlib
from collections import defaultdict
from pathlib import Path

random.seed(42)

# --- Configuration ---
NUM_PAPERS = 3200

CATEGORIES = ["cs.AI", "cs.CL", "cs.CV", "cs.LG", "cs.NE", "stat.ML", "cs.RO", "cs.IR"]
CATEGORY_WEIGHTS_BASE = {
    "cs.AI": 0.12, "cs.CL": 0.18, "cs.CV": 0.20, "cs.LG": 0.22,
    "cs.NE": 0.05, "stat.ML": 0.08, "cs.RO": 0.07, "cs.IR": 0.08,
}

SUBCATEGORIES = {
    "cs.AI": ["planning", "knowledge-representation", "multi-agent", "reasoning", "safety", "alignment"],
    "cs.CL": ["machine-translation", "text-generation", "sentiment-analysis", "question-answering", "summarization", "dialogue"],
    "cs.CV": ["object-detection", "image-segmentation", "image-generation", "video-understanding", "3d-vision", "face-recognition"],
    "cs.LG": ["deep-learning", "optimization", "representation-learning", "meta-learning", "federated-learning", "automl"],
    "cs.NE": ["evolutionary-algorithms", "neural-architecture-search", "neuroevolution", "spiking-networks"],
    "stat.ML": ["bayesian-methods", "kernel-methods", "causal-inference", "time-series", "density-estimation"],
    "cs.RO": ["manipulation", "navigation", "sim-to-real", "human-robot-interaction", "autonomous-driving"],
    "cs.IR": ["recommendation-systems", "search-ranking", "knowledge-graphs", "entity-linking", "dense-retrieval"],
}

VENUES = ["NeurIPS", "ICML", "ICLR", "AAAI", "CVPR", "ACL", "EMNLP", "NAACL", "arXiv-only"]
VENUE_WEIGHTS = {
    "NeurIPS": 0.10, "ICML": 0.08, "ICLR": 0.09, "AAAI": 0.07,
    "CVPR": 0.09, "ACL": 0.06, "EMNLP": 0.05, "NAACL": 0.03,
    "arXiv-only": 0.43,
}

# Category-venue affinity: some venues prefer certain categories
VENUE_CATEGORY_AFFINITY = {
    "CVPR": {"cs.CV": 3.0, "cs.LG": 1.2},
    "ACL": {"cs.CL": 3.0, "cs.IR": 1.5},
    "EMNLP": {"cs.CL": 2.5, "cs.IR": 1.3},
    "NAACL": {"cs.CL": 3.0},
}

PRIMARY_METHODS = [
    "transformer", "cnn", "rnn", "gnn", "diffusion",
    "reinforcement_learning", "bayesian", "ensemble", "other",
]

# Method popularity by year (relative weights)
METHOD_YEAR_WEIGHTS = {
    2018: {"transformer": 0.08, "cnn": 0.30, "rnn": 0.25, "gnn": 0.05, "diffusion": 0.01, "reinforcement_learning": 0.12, "bayesian": 0.07, "ensemble": 0.05, "other": 0.07},
    2019: {"transformer": 0.15, "cnn": 0.25, "rnn": 0.20, "gnn": 0.08, "diffusion": 0.01, "reinforcement_learning": 0.12, "bayesian": 0.06, "ensemble": 0.05, "other": 0.08},
    2020: {"transformer": 0.25, "cnn": 0.20, "rnn": 0.13, "gnn": 0.10, "diffusion": 0.02, "reinforcement_learning": 0.11, "bayesian": 0.06, "ensemble": 0.05, "other": 0.08},
    2021: {"transformer": 0.35, "cnn": 0.15, "rnn": 0.08, "gnn": 0.12, "diffusion": 0.05, "reinforcement_learning": 0.09, "bayesian": 0.05, "ensemble": 0.04, "other": 0.07},
    2022: {"transformer": 0.42, "cnn": 0.10, "rnn": 0.05, "gnn": 0.12, "diffusion": 0.10, "reinforcement_learning": 0.07, "bayesian": 0.04, "ensemble": 0.04, "other": 0.06},
    2023: {"transformer": 0.48, "cnn": 0.07, "rnn": 0.03, "gnn": 0.10, "diffusion": 0.14, "reinforcement_learning": 0.06, "bayesian": 0.03, "ensemble": 0.03, "other": 0.06},
    2024: {"transformer": 0.50, "cnn": 0.05, "rnn": 0.02, "gnn": 0.09, "diffusion": 0.16, "reinforcement_learning": 0.06, "bayesian": 0.03, "ensemble": 0.03, "other": 0.06},
    2025: {"transformer": 0.52, "cnn": 0.04, "rnn": 0.02, "gnn": 0.08, "diffusion": 0.17, "reinforcement_learning": 0.05, "bayesian": 0.03, "ensemble": 0.03, "other": 0.06},
}

# Year distribution: more papers in recent years
YEAR_WEIGHTS = {
    2018: 0.06, 2019: 0.08, 2020: 0.10, 2021: 0.13,
    2022: 0.16, 2023: 0.18, 2024: 0.17, 2025: 0.12,
}

DATASETS_USED = [
    "ImageNet", "COCO", "CIFAR-10", "CIFAR-100", "SQuAD", "GLUE", "SuperGLUE",
    "WMT", "Penn Treebank", "WikiText-103", "OpenWebText", "The Pile",
    "MNIST", "Fashion-MNIST", "CelebA", "LAION-5B", "Common Crawl",
    "BookCorpus", "Cityscapes", "KITTI", "ModelNet40", "ShapeNet",
    "Atari Games", "MuJoCo", "OpenAI Gym", "MovieLens", "Amazon Reviews",
    "Yelp Reviews", "SST-2", "MNLI", "MS MARCO", "Natural Questions",
    "TriviaQA", "HotpotQA", "LSUN", "FFHQ", "CC-12M", "RedPajama",
    "MultiNLI", "SNLI", "AG News", "DBpedia", "custom", "proprietary",
    "multiple", "not specified",
]

# --- Author name pools ---
FIRST_NAMES = [
    "Wei", "Jian", "Yun", "Xin", "Ming", "Chen", "Li", "Hao", "Jun", "Kai",
    "Alex", "James", "David", "Michael", "Robert", "Sarah", "Emily", "Anna",
    "Maria", "Laura", "Siddharth", "Priya", "Aditya", "Neha", "Rahul",
    "Yuki", "Takeshi", "Hiroshi", "Kenji", "Ahmed", "Omar", "Fatima",
    "Pierre", "Marie", "Jean", "Hans", "Stefan", "Marco", "Luca", "Ivan",
    "Olga", "Dmitri", "Andrei", "Carlos", "Diego", "Sofia", "Elena",
    "Thomas", "Daniel", "Benjamin", "Andrew", "Christopher", "Matthew",
    "Zhi", "Tao", "Rui", "Peng", "Feng", "Qiang", "Bo", "Fei",
    "Yann", "Yoshua", "Geoffrey", "Ian", "Ilya", "Dario", "Percy",
    "Sergey", "Ashish", "Noam", "Jakob", "Kaiming", "Ross", "Oriol",
]

LAST_NAMES = [
    "Wang", "Zhang", "Li", "Chen", "Liu", "Yang", "Huang", "Wu", "Zhou", "Sun",
    "Smith", "Johnson", "Brown", "Williams", "Jones", "Taylor", "Wilson",
    "Anderson", "Thomas", "Moore", "Kumar", "Patel", "Singh", "Gupta", "Shah",
    "Tanaka", "Suzuki", "Watanabe", "Yamamoto", "Ali", "Hassan", "Mohamed",
    "Dupont", "Martin", "Bernard", "Mueller", "Schmidt", "Rossi", "Bianchi",
    "Petrov", "Ivanov", "Popov", "Garcia", "Rodriguez", "Martinez",
    "He", "Vaswani", "Shazeer", "Devlin", "Radford", "LeCun", "Bengio",
    "Hinton", "Goodfellow", "Sutskever", "Amodei", "Liang", "Girshick",
    "Vinyals", "Uszkoreit", "Xu", "Zhu", "Lin", "Gao", "Ma", "Ren",
]

# --- Title generation components ---
TITLE_PREFIXES = [
    "Towards", "On", "Rethinking", "Revisiting", "A Survey of",
    "Exploring", "Understanding", "Learning", "Efficient", "Scalable",
    "Robust", "Adaptive", "Self-Supervised", "Semi-Supervised", "Unsupervised",
    "Few-Shot", "Zero-Shot", "Multi-Task", "Multi-Modal", "Cross-Lingual",
    "Federated", "Privacy-Preserving", "Interpretable", "Explainable",
    "End-to-End", "Attention-Based", "Graph-Based", "Memory-Augmented",
    "Curriculum", "Contrastive", "Adversarial", "Unified", "Progressive",
]

TITLE_TOPICS = {
    "cs.AI": [
        "General Intelligence", "Planning under Uncertainty", "Multi-Agent Systems",
        "Knowledge Representation", "Automated Reasoning", "AI Safety",
        "AI Alignment", "Reward Modeling", "Constitutional AI", "Tool Use in LLMs",
        "Chain-of-Thought Reasoning", "Decision Making",
    ],
    "cs.CL": [
        "Language Modeling", "Machine Translation", "Text Summarization",
        "Sentiment Analysis", "Question Answering", "Named Entity Recognition",
        "Dialogue Systems", "Text Generation", "Prompt Engineering",
        "In-Context Learning", "Instruction Tuning", "RLHF",
        "Multilingual Models", "Code Generation", "Document Understanding",
    ],
    "cs.CV": [
        "Object Detection", "Image Segmentation", "Image Generation",
        "Video Understanding", "3D Reconstruction", "Face Recognition",
        "Visual Question Answering", "Scene Understanding", "Depth Estimation",
        "Point Cloud Processing", "Image Super-Resolution", "Style Transfer",
        "Visual Grounding", "Panoptic Segmentation", "Action Recognition",
    ],
    "cs.LG": [
        "Deep Learning Theory", "Optimization Methods", "Representation Learning",
        "Transfer Learning", "Meta-Learning", "Neural Architecture Search",
        "Model Compression", "Knowledge Distillation", "Continual Learning",
        "Out-of-Distribution Detection", "Domain Adaptation", "AutoML",
        "Feature Selection", "Hyperparameter Optimization", "Lottery Ticket Hypothesis",
    ],
    "cs.NE": [
        "Evolutionary Strategies", "Genetic Algorithms", "Neural Architecture Search",
        "Neuroevolution", "Spiking Neural Networks", "Neuromorphic Computing",
    ],
    "stat.ML": [
        "Bayesian Deep Learning", "Gaussian Processes", "Causal Inference",
        "Time Series Forecasting", "Variational Inference", "MCMC Methods",
        "Conformal Prediction", "Kernel Methods", "Density Estimation",
    ],
    "cs.RO": [
        "Robot Manipulation", "Autonomous Navigation", "Sim-to-Real Transfer",
        "Human-Robot Interaction", "Autonomous Driving", "Legged Locomotion",
        "Drone Navigation", "Grasping", "Motion Planning",
    ],
    "cs.IR": [
        "Recommendation Systems", "Dense Retrieval", "Knowledge Graphs",
        "Entity Linking", "Search Ranking", "Collaborative Filtering",
        "Click-Through Rate Prediction", "Session-Based Recommendation",
        "Cross-Modal Retrieval",
    ],
}

TITLE_SUFFIXES = [
    "with Transformers", "using Attention Mechanisms", "via Contrastive Learning",
    "through Self-Supervision", "with Graph Neural Networks", "using Diffusion Models",
    "via Reinforcement Learning", "with Pre-trained Models", "using Knowledge Distillation",
    "for Real-World Applications", "at Scale", "in Low-Resource Settings",
    "with Limited Labels", "under Distribution Shift", "across Domains",
    "with Theoretical Guarantees", "using Large Language Models",
    "via Prompt Tuning", "with Vision Transformers", "using Mixture of Experts",
    "", "", "", "",  # Some papers have no suffix
]

# --- Abstract generation templates ---
ABSTRACT_TEMPLATES = [
    "We propose {method}, a novel approach to {task}. Our method leverages {technique} to achieve state-of-the-art results on {benchmark}. Experiments demonstrate improvements of {improvement} over previous methods. We further analyze the effectiveness of our approach through extensive ablation studies and show its generalization across multiple settings.",
    "This paper introduces {method} for {task}. Unlike prior work that relies on {old_approach}, we design a {technique} framework that directly addresses {challenge}. Evaluation on {benchmark} shows that our approach outperforms existing baselines by a significant margin while requiring fewer computational resources.",
    "Recent advances in {field} have shown promising results for {task}. However, existing methods suffer from {limitation}. We address this by proposing {method}, which combines {technique} with {technique2} to overcome these challenges. Our experiments on {benchmark} demonstrate both improved accuracy and efficiency compared to state-of-the-art approaches.",
    "We present a comprehensive study of {task} in the context of {field}. Through systematic experiments on {benchmark}, we identify key factors that influence model performance and propose {method} as a unified solution. Our approach achieves competitive results while being significantly more parameter-efficient than existing methods.",
    "In this work, we tackle the problem of {task} by introducing {method}. The core idea is to use {technique} to learn robust representations that generalize across {domain}. We validate our approach on {benchmark} and demonstrate substantial improvements. We also provide theoretical analysis supporting our design choices.",
    "We study the problem of {task} and propose {method}, a scalable framework based on {technique}. Our key insight is that {insight}. Extensive experiments on {benchmark} show that our method achieves new state-of-the-art results while maintaining practical inference times. Code and models are publicly available.",
    "This paper presents {method}, designed for efficient {task}. By leveraging {technique} and {technique2}, our approach reduces computational cost by {reduction} while maintaining competitive performance on {benchmark}. We provide detailed analysis of the trade-offs between efficiency and accuracy.",
    "Motivated by recent progress in {field}, we develop {method} for {task}. Our framework integrates {technique} with a novel {component} module that captures {property}. Results on {benchmark} show consistent improvements across all evaluation metrics, establishing a new baseline for future research.",
]

METHODS = [
    "TransFormer++", "DeepMix", "AdaptNet", "FlexiLearn", "NeuralBridge",
    "ContextFlow", "GraphFormer", "DiffuGen", "RobustLearn", "MetaAdapt",
    "UniModel", "CrossModal", "EfficientAI", "ScaleNet", "SmartFuse",
    "DualPath", "HyperNet", "AutoScale", "DeepReason", "FastTune",
    "ProtoLearn", "FlowMatch", "SpectraNet", "DenseFormer", "SparseGPT",
]

TECHNIQUES = [
    "attention mechanisms", "contrastive learning", "knowledge distillation",
    "self-supervised pre-training", "adversarial training", "data augmentation",
    "multi-head attention", "residual connections", "graph convolutions",
    "variational inference", "curriculum learning", "meta-learning",
    "prompt tuning", "LoRA fine-tuning", "mixture of experts",
    "denoising diffusion", "flow matching", "energy-based models",
    "spectral normalization", "gradient checkpointing",
]

FIELDS = [
    "natural language processing", "computer vision", "deep learning",
    "reinforcement learning", "representation learning", "generative modeling",
    "graph learning", "multimodal learning", "transfer learning",
    "neural architecture design", "foundation models", "embodied AI",
]

TASKS = [
    "language understanding", "image classification", "object detection",
    "text generation", "machine translation", "visual reasoning",
    "graph classification", "anomaly detection", "recommendation",
    "speech recognition", "code generation", "robot manipulation",
    "time series forecasting", "image synthesis", "question answering",
    "semantic segmentation", "relation extraction", "document retrieval",
]

BENCHMARKS = [
    "standard benchmarks", "challenging real-world datasets",
    "multiple established benchmarks", "diverse evaluation suites",
    "large-scale datasets", "both synthetic and real-world data",
    "competitive benchmarks", "widely-used evaluation protocols",
]

CHALLENGES = [
    "scalability issues", "distribution shift", "label scarcity",
    "computational overhead", "poor generalization", "catastrophic forgetting",
    "training instability", "data heterogeneity", "domain gap",
]

INSIGHTS = [
    "hierarchical representations capture multi-scale patterns more effectively",
    "sparse attention patterns can approximate dense attention with minimal loss",
    "progressive training schedules significantly improve convergence",
    "auxiliary objectives provide complementary learning signals",
    "architectural simplicity often outperforms complex designs when properly scaled",
    "task-specific adapters preserve pre-trained knowledge while enabling specialization",
]


def weighted_choice(choices_weights):
    """Choose from a dict of {choice: weight}."""
    items = list(choices_weights.keys())
    weights = list(choices_weights.values())
    return random.choices(items, weights=weights, k=1)[0]


def generate_author_name():
    return f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"


def generate_authors(num_authors):
    authors = set()
    while len(authors) < num_authors:
        authors.add(generate_author_name())
    return "; ".join(sorted(authors))


def generate_title(category, method_type):
    prefix = random.choice(TITLE_PREFIXES) if random.random() < 0.6 else ""
    topic = random.choice(TITLE_TOPICS[category])
    suffix = random.choice(TITLE_SUFFIXES)

    if prefix and suffix:
        return f"{prefix} {topic} {suffix}".strip()
    elif prefix:
        return f"{prefix} {topic}".strip()
    elif suffix:
        return f"{topic} {suffix}".strip()
    else:
        return topic


def generate_abstract():
    template = random.choice(ABSTRACT_TEMPLATES)
    abstract = template.format(
        method=random.choice(METHODS),
        task=random.choice(TASKS),
        technique=random.choice(TECHNIQUES),
        technique2=random.choice(TECHNIQUES),
        benchmark=random.choice(BENCHMARKS),
        field=random.choice(FIELDS),
        old_approach=random.choice(TECHNIQUES),
        challenge=random.choice(CHALLENGES),
        limitation=random.choice(CHALLENGES),
        domain="domains",
        insight=random.choice(INSIGHTS),
        improvement=f"{random.uniform(1.5, 12.0):.1f}%",
        reduction=f"{random.randint(20, 70)}%",
        component=random.choice(["attention", "gating", "fusion", "routing", "pooling"]),
        property=random.choice(["long-range dependencies", "local patterns", "semantic relationships", "temporal dynamics"]),
    )
    return abstract


def generate_citation_count(year, venue, is_survey):
    """Power-law citation distribution, adjusted by recency and venue."""
    # Base: power-law distribution
    alpha = 1.5
    x_min = 1
    u = random.random()
    base_citations = int(x_min * (1 - u) ** (-1 / (alpha - 1)))
    base_citations = min(base_citations, 5000)  # cap extreme values

    # Year adjustment: older papers have more citations
    age = 2025 - year
    age_multiplier = 1 + age * 0.4 + (age ** 1.3) * 0.1

    # Venue boost
    venue_multiplier = {
        "NeurIPS": 2.0, "ICML": 1.9, "ICLR": 2.1, "AAAI": 1.4,
        "CVPR": 1.8, "ACL": 1.6, "EMNLP": 1.4, "NAACL": 1.2,
        "arXiv-only": 0.6,
    }
    v_mult = venue_multiplier.get(venue, 1.0)

    # Survey papers get more citations
    survey_mult = 3.0 if is_survey else 1.0

    citations = int(base_citations * age_multiplier * v_mult * survey_mult)

    # Very recent papers (2025) should have few citations
    if year == 2025:
        citations = min(citations, random.randint(0, 30))
    elif year == 2024:
        citations = min(citations, random.randint(0, 150))

    return max(0, citations)


def pick_venue(category):
    """Pick a venue with category affinity."""
    adjusted = dict(VENUE_WEIGHTS)
    for venue, affinities in VENUE_CATEGORY_AFFINITY.items():
        if category in affinities:
            adjusted[venue] = adjusted.get(venue, 0) * affinities[category]
    # Normalize
    total = sum(adjusted.values())
    adjusted = {k: v / total for k, v in adjusted.items()}
    return weighted_choice(adjusted)


def pick_dataset(category, method_type):
    """Pick a dataset relevant to category."""
    category_datasets = {
        "cs.CV": ["ImageNet", "COCO", "CIFAR-10", "CIFAR-100", "CelebA", "LAION-5B", "Cityscapes", "KITTI", "LSUN", "FFHQ", "ModelNet40", "ShapeNet"],
        "cs.CL": ["SQuAD", "GLUE", "SuperGLUE", "WMT", "WikiText-103", "OpenWebText", "The Pile", "BookCorpus", "Common Crawl", "RedPajama", "SST-2", "MNLI"],
        "cs.LG": ["CIFAR-10", "CIFAR-100", "ImageNet", "MNIST", "Fashion-MNIST", "multiple", "custom"],
        "cs.AI": ["OpenAI Gym", "Atari Games", "MuJoCo", "custom", "multiple", "not specified"],
        "cs.NE": ["CIFAR-10", "CIFAR-100", "ImageNet", "MNIST", "custom", "not specified"],
        "stat.ML": ["custom", "proprietary", "multiple", "Penn Treebank", "not specified"],
        "cs.RO": ["MuJoCo", "OpenAI Gym", "custom", "KITTI", "proprietary"],
        "cs.IR": ["MovieLens", "Amazon Reviews", "Yelp Reviews", "MS MARCO", "Natural Questions", "AG News"],
    }
    pool = category_datasets.get(category, DATASETS_USED)
    return random.choice(pool)


def generate_paper(paper_id, year):
    """Generate a single paper record."""
    month = random.randint(1, 12)

    # Category selection (slight year-based shift)
    cat_weights = dict(CATEGORY_WEIGHTS_BASE)
    # CL grows over time (LLM boom)
    if year >= 2022:
        cat_weights["cs.CL"] *= 1.3
        cat_weights["cs.AI"] *= 1.2
    category = weighted_choice(cat_weights)
    subcategory = random.choice(SUBCATEGORIES[category])

    # Method selection based on year
    method_weights = METHOD_YEAR_WEIGHTS[year]
    primary_method = weighted_choice(method_weights)

    # Number of authors: roughly normal distribution centered at 4
    num_authors = max(1, min(15, int(random.gauss(4.2, 2.0))))
    authors = generate_authors(num_authors)

    title = generate_title(category, primary_method)
    abstract = generate_abstract()

    venue = pick_venue(category)
    is_survey = random.random() < 0.04  # ~4% are surveys

    citation_count = generate_citation_count(year, venue, is_survey)

    # has_code: more likely for recent papers and top venues
    code_prob = 0.3 + (year - 2018) * 0.04
    if venue in ["NeurIPS", "ICML", "ICLR", "CVPR"]:
        code_prob += 0.15
    has_code = random.random() < min(code_prob, 0.75)

    dataset_used = pick_dataset(category, primary_method)

    return {
        "paper_id": f"arxiv.{year}{month:02d}.{paper_id:05d}",
        "title": title,
        "abstract": abstract,
        "authors": authors,
        "year": year,
        "month": month,
        "category": category,
        "subcategory": subcategory,
        "citation_count": citation_count,
        "venue": venue,
        "num_authors": num_authors,
        "has_code": has_code,
        "primary_method": primary_method,
        "dataset_used": dataset_used,
        "is_survey": is_survey,
    }


def main():
    print("Generating AI/ML Research Papers dataset...")

    # Distribute papers across years
    papers = []
    paper_id = 1

    for year in range(2018, 2026):
        count = int(NUM_PAPERS * YEAR_WEIGHTS[year])
        for _ in range(count):
            papers.append(generate_paper(paper_id, year))
            paper_id += 1

    # Fill remaining papers randomly across recent years
    while len(papers) < NUM_PAPERS:
        year = random.choices(list(range(2022, 2026)), weights=[0.25, 0.30, 0.28, 0.17], k=1)[0]
        papers.append(generate_paper(paper_id, year))
        paper_id += 1

    # Shuffle to avoid ordering artifacts
    random.shuffle(papers)

    # Write CSV
    fieldnames = [
        "paper_id", "title", "abstract", "authors", "year", "month",
        "category", "subcategory", "citation_count", "venue", "num_authors",
        "has_code", "primary_method", "dataset_used", "is_survey",
    ]

    output_path = Path(__file__).resolve().parent / "ai_research_papers.csv"
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(papers)

    print(f"Generated {len(papers)} papers -> {output_path}")

    # Print basic stats
    from collections import Counter
    year_counts = Counter(p["year"] for p in papers)
    print("\nPapers per year:")
    for y in sorted(year_counts.keys()):
        print(f"  {y}: {year_counts[y]}")

    method_counts = Counter(p["primary_method"] for p in papers)
    print("\nMethod distribution:")
    for m, c in method_counts.most_common():
        print(f"  {m}: {c} ({100*c/len(papers):.1f}%)")

    venue_counts = Counter(p["venue"] for p in papers)
    print("\nVenue distribution:")
    for v, c in venue_counts.most_common():
        print(f"  {v}: {c} ({100*c/len(papers):.1f}%)")


if __name__ == "__main__":
    main()
