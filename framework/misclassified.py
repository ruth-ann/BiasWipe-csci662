import pandas as pd

# Read the CSV file into a DataFrame
df = pd.read_csv('hypothesis.csv') #it contain all entities, filter biased enities->gay, lesbian, transgender, homosexual
keywords=["gay", "lesbian","transgender","homosexual"]
print(df.columns)
# Create an empty list to store the data
result_data = []

# Iterate over the DataFrame
for index, row in df.iterrows():
    text = row['text']
    # idd = row['id']
    actual_label = row['actual_label']
    predicted_label = row['predicted_labels']
    classification = ""
    for keyword in keywords:
        if keyword in text:
            if actual_label != predicted_label:
                # Identify incorrect classifications (false negative or false positive)
                if actual_label and not predicted_label:
                    classification = "FALSE"
                elif not actual_label and predicted_label: #false pos
                    classification = "TRUE"
                
                    # Append data to the result_data list
                    result_data.append({'comment': text, 'is_toxic': actual_label, "prev_predlabel":predicted_label, "keyword":keyword})
                    break
# Create a DataFrame from the result_data list
result_df = pd.DataFrame(result_data)



result_data=[]
 # Iterate over the DataFrame
for index, row in df.iterrows():
    text = row['text']
    # idd = row['id']
    actual_label = row['actual_label']
    predicted_label = row['predicted_labels']
    classification = ""
    # for keyword in keywords:
    if "lesbian" in text:
        if actual_label != predicted_label:
            # Identify incorrect classifications (false negative or false positive)
            if actual_label and not predicted_label:
                classification = "FALSE"
            elif not actual_label and predicted_label: #false pos
                classification = "TRUE"
            
                # Append data to the result_data list
                result_data.append({'comment': text, 'is_toxic': actual_label, "prev_predlabel":predicted_label, "keyword":"lesbian"})
                # break
# Create a DataFrame from the result_data list
result_df = pd.DataFrame(result_data)

# Save the DataFrame to a CSV file
result_df.to_csv('false_pos_lesbian.csv', index=False)
