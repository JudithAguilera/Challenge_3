# CHALLENGE 3: IMAGE CAPTIONING

This project focuses on developing an advanced image captioning system for food images, combining computer vision and natural language processing. We implemented an encoder decoder architecture with attention mechanisms and a transformer based decoder, optimized with beam search to generate precise and coherent captions for food images.
The  goal is to create a robust system capable of generating meaningful and contextually accurate captions, addressing challenges like high visual variability and semantic complexity in food imagery.

**Key Features :**
*  Feature extraction with ResNet-50 --> a pre-trained CNN processes images to extract hierarchical visual features
*   Attention mechanisms --> self attention and cross attention ensure focus on the most relevant image regions while generating captions
*   Transformer based decoder --> ensures fluent and grammatically correct descriptions
*   Beam search decoding --> improves caption quality and diversity by exploring multiple candidate sequences
*   Evaluation metrics --> BLEU and  ROUGE-L are used to validate the quality of the generated captions

**Project workflow:**
1. Dataset Preparation:
*   Split the dataset into training, validation, and testing subsets.
*   Tokenize captions, filter rare words, and convert words to numerical indices.
2. Encoder-Decoder Model:
*   Encoder (ResNet-50): Extracts hierarchical visual features.
*   Attention Mechanisms: Focuses on relevant regions in the images.
*   Decoder (Transformer): Generates captions word by word, aligning with visual features.
3. Training:
*   Optimized with Adam and loss calculated using cross-entropy.
*   Regularized with dropout to prevent overfitting.
*   Trained over 300 epochs with a batch size of 256.
* Beam search decoding ensures coherent and accurate captions.
4. Caption Generation:
* Beam search decoding ensures coherent and accurate captions.
* BLEU and ROUGE-L metrics validate semantic alignment and structural consistency.
5. Evaluation:
* BLEU and ROUGE-L metrics validate semantic alignment and structural consistency.

**Requirements:**
  *   Language: Python 3.8+
  *   Main Libraries:
  *   PyTorch (https://pytorch.org/)
  *   NumPy
  *   Matplotlib
  *   SacreBLEU

**Future improvements:**
*   Design a custom feature extraction network instead of using ResNet-50.
*   Expand the vocabulary with external datasets.
*   Implement hierarchical attention mechanisms.
*   Improve beam search to avoid repetitive or irrelevant terms.
