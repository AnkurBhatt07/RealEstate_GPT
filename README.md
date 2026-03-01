# RealEstate_GPT

This is a real estate chatbot project that helps users search for properties in Bangalore using natural language queries. I built this using LLM models and pandas for filtering the data.

## What This Project Does

You can ask questions like "Show me 2 BHK apartments in Whitefield under 80 lakhs" and the system will understand your query, filter the data, and give you results.

## How It Works

The system works in these steps:
1. User types a query in plain English
2. LLM extracts information like BHK, location, price range from the query
3. Python pandas filters the dataset based on extracted info
4. LLM generates a readable answer with the matching properties

## Dataset

I used the Bangalore real estate dataset which has information about:
- Property location
- Number of bedrooms (BHK)
- Price in lakhs
- Size in square feet
- Bathrooms and balconies
- Availability status
- Society/apartment name

The original dataset had around 13,000 properties but after cleaning and removing duplicates/null values, I got around 7,000 valid properties.

## What I Did

### Data Cleaning (Notebook 1)
- Loaded the raw dataset from banglore.csv
- Removed duplicate entries
- Dropped rows with missing values
- Saved clean data to bangalore_cleaned.csv

### Feature Engineering (Notebook 2)
- Converted the total_sqft column to numeric (it had values like "2100-2200" which I averaged)
- Extracted BHK numbers from the size column using regex
- Created text descriptions for each property
- Generated embeddings using sentence-transformers model
- Built FAISS index for semantic search (though I didn't use this much in the final version)

### Building the Hybrid Model (Notebook 3)
- Used Qwen/Qwen2.5-1.5B-Instruct model to extract filters from user queries
- Wrote functions to filter the dataframe based on extracted filters
- Added sorting functionality (can sort by price or size)
- Made it work with multiple sort criteria
- Generated final responses using the LLM

## Files in the Project

**Notebooks:**
- `1_data_inspection.ipynb` - For exploring and cleaning the dataset
- `2_feature_engineering_and_semantic_indexing.ipynb` - Processing data and creating embeddings
- `3_real_estate_hybrid.ipynb` - Building the hybrid filtering system

**Python Scripts:**
- `hybrid_model.py` - Main model with all functions (filter extraction, filtering, sorting, response generation)
- `app.py` - Streamlit web interface for the hybrid model

**Data:**
- `datasets/banglore.csv` - Original raw dataset
- `cleaned_data/bangalore_cleaned.csv` - Cleaned dataset ready to use

## Technologies Used

- Python (pandas, numpy, re, json)
- Transformers library for loading LLM models
- Sentence-transformers for embeddings
- FAISS for vector search
- Streamlit for the web app
- Qwen2.5-1.5B-Instruct model for understanding queries

## How to Run

1. Make sure you have all the required libraries installed (check requirements.txt)
2. Run the streamlit app:
   ```
   streamlit run app.py
   ```
3. Type your query in the text box and click Search

## Example Queries You Can Try

- "Show me 2 BHK apartments in Whitefield under 80 lakhs"
- "Find 3 BHK properties in Koramangala sorted by price"
- "List 5 apartments in Electronic City between 50-70 lakhs"
- "Show me properties in Hebbal with at least 1200 sqft"

## Things That Work

- Natural language understanding using LLM
- Filtering by BHK, location, price range, and size
- Sorting results by price or area
- Multiple sorting criteria (like sort by price then by size)
- Decent responses from the LLM

## Things I Noticed

The semantic search approach (notebook 2) didn't work as well because it would sometimes return properties from wrong locations. So I switched to using LLM to extract exact filters and then using pandas to filter - this works much better and gives more accurate results.

The model sometimes takes a bit long to respond because it has to run the LLM twice - once for extracting filters and once for generating the response.

## Project Structure

```
RealEstate_GPT/
├── 1_data_inspection.ipynb
├── 2_feature_engineering_and_semantic_indexing.ipynb
├── 3_real_estate_hybrid.ipynb
├── hybrid_model.py
├── app.py
├── README.md
├── requirements.txt
├── datasets/
│   └── banglore.csv
└── cleaned_data/
    └── bangalore_cleaned.csv
```

---

Built as a capstone project for learning about LLMs and real estate data processing.
