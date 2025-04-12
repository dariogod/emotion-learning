# Emotion-Enhanced Natural Language Inference

This project explores whether augmenting NLI data with emotional signals helps language models learn the task more effectively. We use the MNLI (Multi-Genre Natural Language Inference) task as our test case and implement various emotion-aware training strategies.

## Project Overview

The hypothesis is that emotional cues in text may provide valuable signals that help models better understand the relationship between premise and hypothesis in an NLI task. We test this by:

1. Using Anthropic's Claude API to score MNLI samples for emotional content
2. Implementing different training strategies that incorporate these emotional signals
3. Comparing the performance of emotion-aware models against baseline approaches

## Project Structure

```
emotion-learning/
├── GLUE/
│   ├── finetune_glue.py           # Original GLUE finetuning script
│   ├── data_augmentation.py       # Script to augment data with emotion scores
│   ├── emotion_finetune.py        # Emotion-aware training script
│   └── visualization.py           # Visualization utilities for analysis
├── augmented_data/                # Will contain emotion-augmented datasets
├── emotion_models/                # Will contain trained models
├── visualizations/                # Will contain performance visualizations
└── README.md                      # This file
```

## How It Works

### 1. Data Augmentation

We use Anthropic's Claude API to analyze the emotional content of both the premise and hypothesis separately in each MNLI pair. Each component receives:

- **Emotional Intensity**: How emotionally charged the text is (0.0-1.0)
- **Valence**: How positive or negative the emotion is (0.0-1.0)

Additionally, we compute an **Emotional Contrast** score that measures the emotional difference between premise and hypothesis.

The `data_augmentation.py` script handles this process and produces an augmented dataset with these scores.

### 2. Emotion-Aware Training

The `emotion_finetune.py` script implements several ways to incorporate emotional signals:

- **Sample Weighting**: Adjusting the importance of each training example based on its emotional content.
  - Bell Curve: Prioritizes samples with moderate emotional intensity
  - Linear: Higher emotion gets higher weight
  - Inverse: Lower emotion gets higher weight
  - Uniform: Standard training (baseline)

- **Multiple Emotion Targets**: You can use different emotional aspects for weighting:
  - Premise Intensity: Focus on premise emotion
  - Hypothesis Intensity: Focus on hypothesis emotion
  - Contrast: Focus on emotional difference between premise and hypothesis
  - Average: Use the average intensity of both premise and hypothesis

- **Curriculum Learning**: Starting with emotionally simpler examples and gradually introducing more complex ones, with customizable ordering based on different emotional dimensions.

### 3. Visualization and Analysis

The `visualization.py` script generates visualizations to compare:

- Learning curves across different emotion-aware training approaches
- Final performance metrics
- Convergence speed analysis
- Dataset emotion statistics, including premise vs. hypothesis comparisons

## How to Run

### Step 1: Set Up Environment

```bash
# Clone the repository
git clone https://github.com/your-username/emotion-learning.git
cd emotion-learning

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Data Augmentation

```bash
export ANTHROPIC_API_KEY="your-api-key"

python GLUE/data_augmentation.py \
  --model claude-3-sonnet-20240229 \
  --output_dir ./augmented_data \
  --sample_size 1000  # Optional: limit sample size for testing
```

This will create augmented data files in the `augmented_data` directory.

### Step 3: Train Emotion-Aware Models

```bash
# Train with premise-based bell curve weighting
python GLUE/emotion_finetune.py \
  --augmented_data ./augmented_data/mnli_augmented.json \
  --model_name answerdotai/ModernBERT-base \
  --output_dir ./emotion_models/premise_bell_curve \
  --weighting_strategy premise_bell_curve

# Train with hypothesis-based linear weighting
python GLUE/emotion_finetune.py \
  --augmented_data ./augmented_data/mnli_augmented.json \
  --model_name answerdotai/ModernBERT-base \
  --output_dir ./emotion_models/hypothesis_linear \
  --weighting_strategy hypothesis_linear

# Train with contrast-based weighting
python GLUE/emotion_finetune.py \
  --augmented_data ./augmented_data/mnli_augmented.json \
  --model_name answerdotai/ModernBERT-base \
  --output_dir ./emotion_models/contrast_linear \
  --weighting_strategy contrast_linear

# Train with curriculum learning based on premise intensity
python GLUE/emotion_finetune.py \
  --augmented_data ./augmented_data/mnli_augmented.json \
  --model_name answerdotai/ModernBERT-base \
  --output_dir ./emotion_models/curriculum_premise \
  --weighting_strategy uniform \
  --curriculum \
  --curriculum_key premise_intensity

# Train with curriculum learning based on contrast
python GLUE/emotion_finetune.py \
  --augmented_data ./augmented_data/mnli_augmented.json \
  --model_name answerdotai/ModernBERT-base \
  --output_dir ./emotion_models/curriculum_contrast \
  --weighting_strategy uniform \
  --curriculum \
  --curriculum_key contrast

# Train baseline (no emotion weighting)
python GLUE/emotion_finetune.py \
  --augmented_data ./augmented_data/mnli_augmented.json \
  --model_name answerdotai/ModernBERT-base \
  --output_dir ./emotion_models/baseline \
  --weighting_strategy uniform
```

### Step 4: Visualize Results

```bash
python GLUE/visualization.py \
  --experiment_dirs ./emotion_models/premise_bell_curve ./emotion_models/hypothesis_linear ./emotion_models/contrast_linear ./emotion_models/curriculum_premise ./emotion_models/curriculum_contrast ./emotion_models/baseline \
  --experiment_names "Premise Bell Curve" "Hypothesis Linear" "Contrast Linear" "Curriculum (Premise)" "Curriculum (Contrast)" "Baseline" \
  --output_dir ./visualizations
```

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.15+
- Anthropic Python SDK
- Datasets
- Matplotlib
- Seaborn
- pandas
- numpy
- scikit-learn
- tqdm

## Future Work

Potential extensions to this project:

1. Test on other NLI datasets beyond MNLI
2. Explore more sophisticated emotion-aware architectures
3. Analyze error patterns in relation to emotional content
4. Investigate whether specific emotional dimensions (fear, joy, etc.) have different impacts
5. Test with different base models beyond BERT
6. Implement the prompt-based approach mentioned in the paper (adding emotion tags to input text)

## License

This project is licensed under the MIT License - see the LICENSE file for details.
