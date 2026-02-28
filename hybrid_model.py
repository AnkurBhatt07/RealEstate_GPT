import pandas as pd 
import re
import json
from transformers import pipeline
import os 
os.environ['HF_HOME'] = "D:/hf_cache"

df = pd.read_csv("cleaned_data/bangalore_cleaned.csv")


def convert_sqft(x):
    try:
        if '-' in str(x):
            a,b = x.split('-')
            return (float(a) + float(b))/2
        return float(x)
    except:
        return None 
    
df['total_sqft'] = df['total_sqft'].apply(convert_sqft)


df = df.dropna(subset=['total_sqft'])


def extract_bhk(x):
    if pd.isna(x):
        return None
    x = x.lower().strip()

    match = re.search(r'(\d+)\s*(bhk|bedroom|rk)',x)

    if match:
        return int(match.group(1))
    
    return None

df['bhk'] = df['size'].apply(extract_bhk) 


df = df.dropna(subset=['bhk'])


model_id = 'Qwen/Qwen2.5-1.5B-Instruct'

gen = pipeline(
    task = 'text-generation',
    model = model_id,
    device = 0,
    model_kwargs = {'cache_dir': "D:/hf_cache"}
)




def extract_filters_llm(query):

    prompt = f"""
        Extract filters from the real estate query.
        You MUST return ONLY a valid JSON object.
        Do not include any text before or after the JSON.
        Do not include explanations, comments, or code.
        The output must start with '{' and end with '}'.

        Rules:
- Use double quotes for ALL keys and string values
- Do NOT use single quotes
- Use null instead of None
- Do NOT include trailing commas
- Ensure valid JSON that can be parsed by json.loads()


        Return ONLY valid JSON with these fields:
        - bhk (int or null)
        - location (string or null)
        - max_price (float or null , in lakhs) 
        - min_price (float or null , in lakhs) 
        - min_sqft (float or null , in sqft)
        - max_sqft (float or null , in sqft) 
        - sort_by ("price" or "sqft")
        - sort_order ("asc" or "desc")
        - top_k (int) - number of listings to retrieve
        
        INSTRUCTION:
        1. DO NOT RETURN ANY EXPLANATION. ONLY RETURN THE JSON.DO NOT RETURN ANY CODE FOR QUERY MANIPULATION TO JSON. ONLY RETURN THE JSON.

        2. IF top_k IS NOT MENTIONED IN THE QUERY , DEFAULT IT TO 5

        3. top_k should be extracted from words like:
            - top 5 
            - first 10  
            - Show 3 properties 
            - list 2 flats

        4. sort_by can be a list if multiple sort criteria are mentioned in the query. 
            eg - ["price" , "total_sqft"] or ["total_sqft", "price"]

        5. sort_order can also be a list if multiple sort criteria are mentioned in the query. 
            The order of sort_order should correspond to the order of sort_by.
            eg - ["asc", "desc"] or ["desc", "asc"]

        6. Do not generate extra Query-Output pairs in the output. Only process the given query and generate output for it.
            
    Examples:

    Query:
    Show me top 5 2-BHK apartments in Whitefield under 80 lakhs sorted by price low to high
    Output: {{"bhk": 2, "location": "Whitefield", "max_price": 80.0, "min_price": null, "min_sqft": null, "max_sqft": null, "sort_by": "price", "sort_order": "asc", "top_k": 5}}

    Query:
    Give me five 3-BHK flats in Koramangala under 1.5 crores
    Output: {{"bhk": 3, "location": "Koramangala", "max_price": 150.0, "min_price": null, "min_sqft": null, "max_sqft": null, "sort_by": null, "sort_order": null, "top_k": 5}}

    Query:
    List first ten apartments in Indiranagar between 1000 sqft and 2000 sqft sorted by sqft low to high
    Output: {{"bhk": null, "location": "Indiranagar", "max_price": null, "min_price": null, "min_sqft": 1000.0, "max_sqft": 2000.0, "sort_by": "sqft", "sort_order": "asc", "top_k": 10}}

    Query:
    Show me six 4-BHK villas in Sarjapur Road above 2 crores
    Output: {{"bhk": 4, "location": "Sarjapur Road", "max_price": null, "min_price": 200.0, "min_sqft": null, "max_sqft": null, "sort_by": null, "sort_order": null, "top_k": 6}}

    Query:
    Show seven 1-BHK apartments in Electronic City under 50 lakhs sorted by price high to low
    Output: {{"bhk": 1, "location": "Electronic City", "max_price": 50.0, "min_price": null, "min_sqft": null, "max_sqft": null, "sort_by": "price", "sort_order": "desc", "top_k": 7}}

    Query:
    Find five apartments in Hebbal with minimum 1200 sqft area
    Output: {{"bhk": null, "location": "Hebbal", "max_price": null, "min_price": null, "min_sqft": 1200.0, "max_sqft": null, "sort_by": null, "sort_order": null, "top_k": 5}}

    Query:
    List eight 2-BHK flats in Marathahalli between 60 lakhs and 90 lakhs
    Output: {{"bhk": 2, "location": "Marathahalli", "max_price": 90.0, "min_price": 60.0, "min_sqft": null, "max_sqft": null, "sort_by": null, "sort_order": null, "top_k": 8}}

    Query:
    Show top four 3-BHK apartments in Whitefield above 1 acre sorted by price high to low
    Output: {{"bhk": 3, "location": "Whitefield", "max_price": null, "min_price": null, "min_sqft": 43560.0, "max_sqft": null, "sort_by": "price", "sort_order": "desc", "top_k": 4}}

    # 1 acre = 43560 sqft

    Query:
    Find six 2-BHK flats in Koramangala between 100 sqm and 200 sqm
    Output: {{"bhk": 2, "location": "Koramangala", "max_price": null, "min_price": null, "min_sqft": 1076.0, "max_sqft": 2152.0, "sort_by": null, "sort_order": null, "top_k": 6}}

    # 1 sqm = 10.764 sqft

    Query:
    Show nine villas in Indiranagar above 500 sq yards sorted by sqft low to high
    Output: {{"bhk": null, "location": "Indiranagar", "max_price": null, "min_price": null, "min_sqft": 4500.0, "max_sqft": null, "sort_by": "sqft", "sort_order": "asc", "top_k": 9}}

    # 1 square yard = 9 sqft → 500 sq yards = 4,500 sqft

    Query:
    Show me seven 3-BHK apartments in Whitefield under 3 crores sorted by sqft high to low
    Output: {{"bhk": 3, "location": "Whitefield", "max_price": 300.0, "min_price": null, "min_sqft": null, "max_sqft": null, "sort_by": "sqft", "sort_order": "desc", "top_k": 7}}

    Query:
    Show top five 2-BHK apartments in Whitefield sorted by price low to high and sqft high to low
    Output: {{
    "bhk": 2,
    "location": "Whitefield",
    "max_price": null,
    "min_price": null,
    "min_sqft": null,
    "max_sqft": null,
    "sort_by": ["price", "total_sqft"],
    "sort_order": ["asc", "desc"],
    "top_k": 5
    }}

    Query:
    List first four 3-BHK flats in Koramangala under 2 crores sorted first by sqft low to high then by price high to low
    Output: {{
    "bhk": 3,
    "location": "Koramangala",
    "max_price": 200.0,
    "min_price": null,
    "min_sqft": null,
    "max_sqft": null,
    "sort_by": ["total_sqft", "price"],
    "sort_order": ["asc", "desc"],
    "top_k": 4
    }}

    Query:
    Find six apartments in Indiranagar between 1000 sqft and 2000 sqft sorted by price high to low and sqft low to high
    Output: {{
    "bhk": null,
    "location": "Indiranagar",
    "max_price": null,
    "min_price": null,
    "min_sqft": 1000.0,
    "max_sqft": 2000.0,
    "sort_by": ["price", "total_sqft"],
    "sort_order": ["desc", "asc"],
    "top_k": 6
    }}


        Now Process:
        Query: {query}

Before returning, verify:
- Output is valid JSON
- All keys are present
- No extra text is included
        """ 
    out = gen(prompt , max_new_tokens = 200 , do_sample = False ,return_full_text = False , temperature = 0.0)

    text = out[0]['generated_text']

    # # For debugging purpose
    # print("LLM filters Extraction Output:\n", text)


    try:
        match = text.replace("\n" , " ").replace("None" , "null")
        
        start_index = match.find('{')
        end_index = match.find('}')
        output = match[start_index:end_index+1]
        
        filters_dict = json.loads(output)
        return filters_dict

    except Exception as e:
        print("Error in extracting filters:", e)
        return None

# This functions returns filters in python dictionary format

def apply_filters(df , filters):

    result = df.copy()

    if filters.get('bhk'):
        result = result[result['bhk'] == filters['bhk']]

    if filters.get('location'):
        result = result[result['location'].str.lower() == filters['location'].lower()]
    if filters.get('max_price'):
        result = result[result['price'] <= filters['max_price']] 

    if filters.get('min_price'):
        result = result[result['price'] >= filters['min_price']]

    if filters.get('min_sqft'):
        result = result[result['total_sqft'] >= filters['min_sqft']]

    if filters.get('max_sqft'):
        result = result[result['total_sqft'] <= filters['max_sqft']]

    return result 


def apply_sorting(df, filters):
    sort_by = filters.get('sort_by')
    order = filters.get("sort_order" , 'asc')

    # default 
    ascending = True if order == 'asc' else False 

    # MULTI SORTING SUPPORT:
    if isinstance(sort_by , list):
        ascending_list = []
        for col, ord in zip(sort_by , filters.get('sort_order',[])):
            ascending_list.append(True if ord == 'asc' else False)
        df = df.sort_values(by= sort_by , ascending = ascending_list)

    else:
        if sort_by in ['price', 'total_sqft']:
            df = df.sort_values(by=sort_by, ascending=ascending)

    return df




# Normal answer query function without memory implementation

def answer_query(query ):
    filters = extract_filters_llm(query)
    top_k = filters.get('top_k' , 5)
    filtered = apply_filters(df , filters)

    sorted_df = apply_sorting(filtered , filters)

    top_df = sorted_df.head(top_k)

   
    context = "\n".join([
        f"""
Properties {i+1}:
- BHK: {row['bhk']}
- Location: {row['location']}
- Price: {row['price']} lakh
- Size : {row['total_sqft']} sqft
- Bathrooms: {row['bath']}
- Balcony: {row['balcony']}
- Area Type: {row['area_type']}
- Availability: {row['availability']}
- Society: {row['society']}
"""
for i , (_ , row) in enumerate(top_df.iterrows())
])
    

    prompt = f"""
    
    You are a real estate assistant.

    Use ONLY the given properties.

    Properties:
    {context}

    User Query: {query}

    Answer in bullet points:

    """

    out = gen(prompt , max_new_tokens = 500 , do_sample = False , return_full_text = False)

    return out[0]['generated_text']
    

