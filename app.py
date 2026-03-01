import streamlit as st 

from hybrid_model import answer_query

st.set_page_config(page_title = "Real Estate GPT" , layout = "centered")

st.title("Real Estate GPT")
st.write("Ask property queries like:")
st.write("Example : Show me three 2-BHK properties in Whitefield with price under 80 lakhs and sort them by size descending")

query = st.text_input("Enter your query here")

if st.button("Search"):
    if query.strip()=="":
        st.warning("Please enter a query.")
    else:
        with st.spinner("Processing your query..."):
            try:
                answer = answer_query(query)
                st.success("Here are the results:")
                st.write(answer)
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")