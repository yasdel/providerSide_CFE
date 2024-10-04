# providerSide_CFE

# Data Enrichment and Auto-Tagging Using Large Language Models

## Python Code: MovieLens Data Augmentation and Auto-Tagging

The code [`TOIS24-pCFE_MovieLens_DataAugmentation.ipynb`](https://drive.google.com/file/d/1ycek-02Mr2g2epkMF8nRcAFhd2_D48Aq/view?usp=sharing) serves two primary functions:

1. **Data Enrichment**:  
   It processes item profiles (e.g., from **MovieLens**) that include information such as:
   - **Title**
   - **Genre**
   - **Tags**
   
   Using a **Large Language Model (LLM)**, it generates enriched descriptions based on the combination of `<title, genre, (tag)>` information. This makes the item profiles more detailed and informative, supporting better recommendations and analysis.

   > **Example:**  
   > For a movie with the title *Inception*, genre *Science Fiction*, and tag *dream manipulation*, the LLM generates a richer description that provides more insight into the item.

2. **Auto-Tagging**:  
   In the second part of the code, **auto-tagging** is performed using Natural Language Processing (NLP) techniques. After the LLM enriches the descriptions, the code:

   - Cleans and tokenizes the enriched descriptions.
   - Filters for nouns and adjectives using **NLTK**'s part-of-speech tagging.
   - Applies **TF-IDF (Term Frequency-Inverse Document Frequency)** to identify the most relevant tags from the enriched descriptions.
   
   The goal is to automatically generate tags that represent key themes or attributes of each item based on the enriched text.

   > **Example Process:**
   > - Tokenize the description.
   > - Remove stopwords and punctuation.
   > - Filter out only **nouns** and **adjectives**.
   > - Apply **TF-IDF** to extract top tags for each item.

   Below is a snippet from the auto-tagging code:

   ```python
   # Apply the tokenizer and filter to the enriched_description column
   df_items_enriched['filtered_tokens'] = df_items_enriched['enriched_description'].apply(tokenize_clean_and_filter)

   # Create the TF-IDF vectorizer
   vectorizer = TfidfVectorizer()
   tfidf_matrix = vectorizer.fit_transform(df_items_enriched['filtered_text'])

   # Get the feature names (words) from the TF-IDF model as potential tags
   feature_names = vectorizer.get_feature_names_out()

   # Function to get top N tags based on TF-IDF scores
   def get_top_tags(tfidf_row, feature_names, top_n=5):
       sorted_indices = tfidf_row.argsort()[::-1][:top_n]
       return [feature_names[i] for i in sorted_indices]

   # Apply the tagging function to each row in the DataFrame
   df_items_enriched['generated_tags'] = [get_top_tags(tfidf_row, feature_names)
                                              for tfidf_row in tfidf_matrix.toarray()]
