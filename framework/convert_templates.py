import pandas as pd
import re
import unicodedata

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

df = pd.read_csv("/home/exouser/data/unintended-ml-bias-analysis/sentence_templates/es-es_sentence_templates.csv")

with open("/home/exouser/data/unintended-ml-bias-analysis/identity_terms_es.txt", "r", encoding="utf-8") as f:
    identity_terms = [line.strip() for line in f if line.strip()]

# Sort by length descending to prioritize multi-word terms
identity_terms_sorted = sorted(identity_terms, key=len, reverse=True)

# Correct the phrases first
df["comment"] = df["phrase"].apply(lambda x: correct_phrase_strict(x, identity_terms_sorted))

# Extract keywords from corrected phrases
df["keyword"] = df["comment"].apply(lambda x: extract_identity_from_corrected(x, identity_terms_sorted))

# Convert toxicity to binary
df["is_toxic"] = df["toxicity"].apply(lambda x: 1 if x == "toxic" else 0)

# Select and order columns
df_out = df[["comment", "keyword", "is_toxic"]]

# Save to CSV
df_out.to_csv("/home/exouser/data/unintended-ml-bias-analysis/spanish_entities.csv", index=False)

print(f"Processed {len(df_out)} rows. Saved to spanish_entities.csv")


# import pandas as pd
# import re
# import unicodedata


# # Load CSV
# df = pd.read_csv("/home/exouser/data/unintended-ml-bias-analysis/sentence_templates/es-es_sentence_templates.csv")

# with open("/home/exouser/data/unintended-ml-bias-analysis/identity_terms_es.txt", "r") as f:
#     identity_terms = [line.strip().lower() for line in f if line.strip()]

# identity_terms_sorted = sorted(identity_terms, key=len, reverse=True)

# # def extract_identity(phrase):
# #     """
# #     ENGLISH VERSION
# #     Gets given identity term/occupations (in identity_terms.txt file) for each sentence
# #     """
# #     phrase_lower = phrase.lower()
# #     print(f"\nChecking phrase: '{phrase}'")

# #     for identity in identity_terms_sorted:
# #         identity_lower = identity.lower()
# #         if re.search(r'\b' + re.escape(identity_lower) + r'\b', phrase_lower):
# #             return identity  # return the full matching term
# #     return ""


# def remove_accents(text):
#     return ''.join(c for c in unicodedata.normalize('NFD', text)
#                    if unicodedata.category(c) != 'Mn')

# def extract_identity(phrase):
#     """
#     SPANISH VERSION
#     Gets given identity term/occupations (in identity_terms_es.txt file) for each sentence
#     """
#     phrase_norm = remove_accents(phrase.lower())

#     for identity in identity_terms_sorted:
#         identity_norm = remove_accents(identity.lower())
#         if re.search(r'\b' + re.escape(identity_norm) + r'\b', phrase_norm):
#             return identity  # return full original term
    
#     return ""

# df["keyword"] = df["phrase"].apply(extract_identity)

# # # Comment is simply the full phrase
# df["comment"] = df["phrase"]

# # # Convert toxicity into binary label
# df["is_toxic"] = df["toxicity"].apply(lambda x: 1 if x == "toxic" else 0)

# # # Select and order columns
# df_out = df[["comment", "keyword", "is_toxic"]]

# # # Save to new CSV
# df_out.to_csv("/home/exouser/data/unintended-ml-bias-analysis/spanish_entities.csv", index=False)

