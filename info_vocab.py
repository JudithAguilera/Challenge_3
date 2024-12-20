import json
from collections import Counter

def describe_vocabulary(vocab_path, dataset_csv):
    # Load vocabulary
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)

    itos = {int(k): v for k, v in vocab_data["itos"].items()}
    stoi = vocab_data["stoi"]

    vocab_size = len(itos)
    pad_token = itos[0]
    sos_token = itos[1]
    eos_token = itos[2]
    unk_token = itos[3]

    # Analyze the dataset for frequency distribution
    import pandas as pd
    df = pd.read_csv(dataset_csv)
    captions = df['Title'].fillna('').tolist()

    word_counts = Counter()
    for caption in captions:
        word_counts.update(caption.split())

    # Words included in vocabulary
    included_words = [word for word in word_counts if word in stoi]
    excluded_words = [word for word in word_counts if word not in stoi]

    included_count = sum(word_counts[word] for word in included_words)
    excluded_count = sum(word_counts[word] for word in excluded_words)

    # Description text
    description = f"""
    Vocabulary Description:
    -----------------------
    - Vocabulary Size: {vocab_size}
    - Special Tokens:
        - <PAD>: {pad_token}
        - <SOS>: {sos_token}
        - <EOS>: {eos_token}
        - <UNK>: {unk_token}
    
    Dataset Analysis:
    -----------------
    - Total Words in Captions: {sum(word_counts.values())}
    - Unique Words in Captions: {len(word_counts)}
    - Words Included in Vocabulary: {len(included_words)}
    - Words Excluded from Vocabulary: {len(excluded_words)}
    - Percentage of Words Covered by Vocabulary: {included_count / (included_count + excluded_count) * 100:.2f}%
    
    Examples of Included Words: {included_words[:10]}
    Examples of Excluded Words: {excluded_words[:10]}
    """

    # Save description to file
    output_path = "vocabulary_description.txt"
    with open(output_path, 'w') as file:
        file.write(description)

    print(f"Vocabulary description written to {output_path}")

# Example usage
describe_vocabulary(vocab_path="vocab.json", dataset_csv="train.csv")
