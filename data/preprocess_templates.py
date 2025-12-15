import pandas as pd
import re
import unicodedata
import argparse

def remove_accents(text):
    """Removes accents from a string for normalization"""
    return ''.join(c for c in unicodedata.normalize('NFD', text)
                   if unicodedata.category(c) != 'Mn')

def correct_phrase_strict(phrase, identity_terms_sorted):
    """
    Correct misaccented phrases (spanish) so that only exact matches to identity terms are replaced
    with the correctly accented version
    """
    words = phrase.split()
    corrected_words = []

    for word in words:
        word_norm = remove_accents(word.lower())
        replaced = False
        for identity in identity_terms_sorted:
            identity_norm = remove_accents(identity.lower())
            if word_norm == identity_norm:  # exact normalized match
                corrected_words.append(identity)
                replaced = True
                break
        if not replaced:
            corrected_words.append(word)
    
    return " ".join(corrected_words)

def extract_identity_from_corrected(phrase, identity_terms_sorted):
    """
    Extracts the first matching identity term from a corrected phrase
    """
    phrase_norm = remove_accents(phrase.lower())
    for identity in identity_terms_sorted:
        identity_norm = remove_accents(identity.lower())
        if re.search(r'\b' + re.escape(identity_norm) + r'\b', phrase_norm):
            return identity
    return ""

def main(input_csv, output_csv, identity_terms_path):
    df = pd.read_csv(input_csv)
    print("Input Dataframe Loaded")

    with open(identity_terms_path, "r", encoding="utf-8") as f:
        identity_terms = [line.strip() for line in f if line.strip()]

    # Sort by length descending to prioritize multi-word terms
    identity_terms_sorted = sorted(identity_terms, key=len, reverse=True)

    # Correct the phrases first
    df["comment"] = df["phrase"].apply(lambda x: correct_phrase_strict(x, identity_terms_sorted))

    # Extract keywords from corrected phrases
    df["keyword"] = df["comment"].apply(lambda x: extract_identity_from_corrected(x, identity_terms_sorted))
    print("Keywords Extracted")

    # Convert toxicity to binary
    df["is_toxic"] = df["toxicity"].apply(lambda x: 1 if x == "toxic" else 0)
    print("Toxicity Converted to Binary")

    # Select and order columns
    df_out = df[["comment", "keyword", "is_toxic"]]

    # Save to CSV
    df_out.to_csv(output_csv, index=False)

    print(f"Processed {len(df_out)} rows. Saved to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Spanish sentence templates")
    parser.add_argument("--input_path", required=True, help="Path to input CSV file")
    parser.add_argument("--output_path", required=True, help="Path to output CSV file")
    parser.add_argument("--identity_terms_path", required=True, help="Path to identity terms TXT file")

    args = parser.parse_args()
    main(args.input_path, args.output_path, args.identity_terms_path)