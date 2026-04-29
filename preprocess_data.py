import json
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK models
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')

def create_preprocessed_file(json_file_path, output_csv_path):
    print("Loading data...")
    with open(json_file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    df = pd.DataFrame(data)
    
    # Filter for MORLY org_code
    morly_df = df[df['org_code'] == 'MORLY'].copy()
    print(f"Total MORLY records found: {len(morly_df)}")

    # 1. Standard English Stopwords
    english_stops = set(stopwords.words('english'))

    # 2. Refined Hinglish Stopwords (Roman script)
    hinglish_stops = {
        # Connectors & Prepositions
        'hai', 'hain', 'ka', 'ki', 'ke', 'ko', 'se', 'par', 'pe', 'mein', 'me', 'aur', 'ya', 
        'toh', 'to', 'bhi', 'hi', 'ne', 'tak', 'liye', 'lekin', 'kiya', 'ji', 'sir', 'madam',
        'please', 'pls', 'help', 'madad',
        # Tense Markers & Auxiliaries
        'tha', 'thi', 'the', 'ho', 'raha', 'rahi', 'rahe', 'gaya', 'gayi', 'gaye', 'hua', 
        'hui', 'huye', 'hoon', 'hona', 'hota', 'hoti', 'hote', 'aaya', 'aayi', 'aate',
        # Pronouns & Pointers
        'mujhe', 'maine', 'mera', 'meri', 'mere', 'hum', 'humein', 'aap', 'aapka', 'aapki', 
        'aapke', 'yeh', 'woh', 'is', 'us', 'iska', 'uska', 'unhe', 'unka', 'hume', 'itna',
        # Interrogatives & Quantifiers
        'kya', 'kyun', 'kab', 'kahan', 'kaise', 'kitna', 'kitne', 'kuch', 'koi', 'kabhi', 
        'sab', 'baaki', 'kaafi', 'bilkul', 'islye',
        # Generic Verb Fillers
        'kar', 'karo', 'karein', 'karna', 'karne', 'karke', 'diya', 'diye', 'de', 'le', 
        'liya', 'liye', 'aa', 'ja'
    }

    # Union of both lists
    all_stops = english_stops.union(hinglish_stops)

    def clean_text(text):
        if not isinstance(text, str): 
            return ""
        
        # Lowercase
        text = text.lower()
        
        # Remove punctuation, numbers, and special characters
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter out stopwords and short noise (length <= 2)
        tokens = [t for t in tokens if t not in all_stops and len(t) > 2]
        
        return " ".join(tokens)

    print("Cleaning text using unified stopword list...")
    morly_df['cleaned_text'] = morly_df['subject_content_text'].apply(clean_text)

    # Drop records that became empty after cleaning
    processed_df = morly_df[morly_df['cleaned_text'] != ""].reset_index(drop=True)
    
    # Save the necessary columns to a CSV
    processed_df[['registration_no', 'subject_content_text', 'cleaned_text']].to_csv(output_csv_path, index=False)
    print(f"Preprocessing complete. Saved {len(processed_df)} records to {output_csv_path}")

# Run the script
if __name__ == "__main__":
    create_preprocessed_file('no_pii_grievance_extension.json', 'preprocessed_grievances.csv')