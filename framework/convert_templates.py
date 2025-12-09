import pandas as pd

# Load CSV
df = pd.read_csv("/home/exouser/data/unintended-ml-bias-analysis/sentence_templates/es-es_sentence_templates.csv")

with open("/home/exouser/data/unintended-ml-bias-analysis/identity_terms_es.txt", "r") as f:
    identity_terms = [line.strip().lower() for line in f if line.strip()]

def extract_identity(phrase):
    """
    Gets given identity term/occupations (in identity_terms.txt file) for each sentence
    """
    phrase_lower = phrase.lower()
    for identity in identity_terms:
        if identity in phrase_lower:
            return identity  
    print(phrase)
    return ""

df["keyword"] = df["phrase"].apply(extract_identity)

# # Extract keyword (last word in the phrase)
# df["keyword"] = df["phrase"].apply(lambda x: x.split()[-1])

# # Comment is simply the full phrase
df["comment"] = df["phrase"]

# # Convert toxicity into binary label
df["is_toxic"] = df["toxicity"].apply(lambda x: 1 if x == "toxic" else 0)

# # Select and order columns
df_out = df[["comment", "keyword", "is_toxic"]]

# # Save to new CSV
df_out.to_csv("/home/exouser/data/unintended-ml-bias-analysis/spanish_entities.csv", index=False)

