# providerSide_CFE

# Data Enrichment Using Large Language Models

## Python Code: MovieLens Data Augmentation

The code [`TOIS24-pCFE_MovieLens_DataAugmentation.ipynb`](https://drive.google.com/file/d/1ycek-02Mr2g2epkMF8nRcAFhd2_D48Aq/view?usp=sharing) is responsible for **data enrichment**. It takes item profiles (e.g., from **MovieLens**) that include information such as:

- **Title**
- **Genre**
- **Tags**

The code then enriches these profiles using a **Large Language Model (LLM)** (e.g., ChatGPT-3.5Turbo). This enrichment process involves generating descriptive content based on the combination of `<title, genre, (tag)>` information.

The goal of this process is to enhance item profiles, making them more detailed and informative, which in turn supports better recommendations and analysis.

> **Example:**  
> For a movie with the title *Inception*, genre *Science Fiction*, and tag *dream manipulation*, the LLM generates a richer description that provides more insight into the item.

