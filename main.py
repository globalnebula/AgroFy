import streamlit as st
import pandas as pd
import requests
import numpy as np
import cv2
from PIL import Image
import io
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

groq_api_key = "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"

# Initialize Groq client
def initialize_groq_client(api_key):
    client = Groq(api_key=api_key)
    return client

import os
import PyPDF2

@st.cache_resource
def load_knowledge_base_and_embeddings():
    # Load the knowledge base (CSV file and PDF file)
    csv_file_path = 'agriKnow.csv'
    pdf_file_path = 'agriKnow.pdf'
    
    # Initialize DataFrame and PDF content
    df = None
    pdf_content = None

    # Check if the CSV file exists
    if os.path.exists(csv_file_path):
        df = pd.read_csv(csv_file_path)
        
        # Create a 'content' column from existing columns, if required
        df['content'] = df.apply(lambda row: f"{row['Crop Name']} - {row['Notes']}", axis=1)
    else:
        st.error("CSV file not found.")
    
    # Check if the PDF file exists
    if os.path.exists(pdf_file_path):
        # Extract text from the PDF file
        pdf_content = extract_pdf_content(pdf_file_path)
    else:
        st.error("PDF file not found.")
    
    # Load embeddings (if DataFrame is not None)
    if df is not None and 'content' in df.columns:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Create document embeddings
        document_embeddings = model.encode(df['content'].tolist())
        
        # Build a FAISS index for fast retrieval
        index = faiss.IndexFlatL2(document_embeddings.shape[1])  # L2 distance
        index.add(np.array(document_embeddings))
        
        return df, index, model, pdf_content
    
    return df, None, None, pdf_content

def extract_pdf_content(pdf_file_path):
    """Extract text content from the PDF file."""
    content = ""
    with open(pdf_file_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        for page in reader.pages:
            content += page.extract_text() + "\n"
    return content.strip()


df, index, model, pdf_content = load_knowledge_base_and_embeddings()


if pdf_content is not None:
    st.markdown("### Knowledge Base PDF")
    st.write("You can download the PDF knowledge base for RAG model [here](agriKnow.pdf).")


def retrieve_documents(query, df, index, model, top_k=3):
    query_embedding = model.encode([query])
    
    # Search in FAISS index
    distances, indices = index.search(np.array(query_embedding), top_k)
    
    # Return the top_k documents
    retrieved_docs = df.iloc[indices[0]]
    
    return retrieved_docs

# Function to interact with Groq AI for agriculture-related queries
def fetch_agriculture_advice(user_query, client, location):
    openweather_api_key = "8b7d1ed40b332b731beb9ee190eab6d9"
    precipitation, temperature = fetch_weather_data(openweather_api_key, location)
    context = f"The current precipitation is {precipitation} mm and the temperature is {temperature} °C. at {location}"
    query = f"{user_query} in context of the the conditions {context}."
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant designed to support farmers by providing accurate and timely information related to agriculture."
            },
            {
                "role": "user",
                "content": query
            }
        ],
        temperature=1,
        max_tokens=1024,
        top_p=1,
        stream=True,
        stop=None,
    )

    response = ""
    for chunk in completion:
        response += chunk.choices[0].delta.content or ""
    
    return response

# Function to interact with Groq AI (LLM) for a direct response
def fetch_llm_answer(user_query, client):
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": "You are an AI assistant designed to support farmers by providing accurate and timely information related to agriculture. You understand various aspects of farming, including crop management, soil health, weather forecasting, pest control, irrigation techniques, market prices, government schemes, and modern farming technologies."
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        temperature=1,
        max_tokens=512,
        top_p=1,
        stream=True,
        stop=None,
    )

    llm_response = ""
    for chunk in completion:
        llm_response += chunk.choices[0].delta.content or ""
    
    return llm_response

# Function to fetch Groq AI's answer after retrieving context with RAG
def fetch_rag_answer(user_query, client, retrieved_docs):
    # Combine retrieved documents as context
    context = "\n\n".join(retrieved_docs['content'].tolist())
    
    completion = client.chat.completions.create(
        model="llama3-8b-8192",
        messages=[
            {
                "role": "system",
                "content": f"You are an AI assistant designed to support farmers by providing accurate and timely information related to agriculture. You understand various aspects of farming, including crop management, soil health. Here is some relevant context for the user's query: {context}"
            },
            {
                "role": "user",
                "content": user_query
            }
        ],
        temperature=1,
        max_tokens=512,
        top_p=1,
        stream=True,
        stop=None,
    )

    rag_response = ""
    for chunk in completion:
        rag_response += chunk.choices[0].delta.content or ""
    
    return rag_response

# Fusion function: Combine LLM's and RAG's answers
def fusion_response(llm_answer, rag_answer):
    # Simple fusion strategy: Combine both responses
    final_response = f"LLM Answer:\n{llm_answer}\n\nRAG-Enhanced Answer:\n{rag_answer}"
    
    return final_response

# Function to fetch precipitation and temperature data from OpenWeatherMap API
def fetch_weather_data(api_key, location):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        precipitation = data.get('rain', {}).get('1h', 0)  # Precipitation in last hour
        temperature = data.get('main', {}).get('temp', 0)  # Current temperature
        return precipitation, temperature
    else:
        return None, None


def fetch_ndvi_image(nasa_image_url):
    response = requests.get(nasa_image_url)
    if response.status_code == 200:
        return Image.open(io.BytesIO(response.content))
    else:
        return None

# Function to process NDVI image
def process_ndvi_image(image):
    # Convert image to numpy array
    image_array = np.array(image)
    
    # Convert to grayscale
    gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

    # Thresholding to create binary NDVI image
    _, binary_ndvi = cv2.threshold(gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    return binary_ndvi


st.markdown("""
    <style>
    /* Adjust all headers globally */
    h1 {
        font-size: 40px;
        color: #fff;
        text-align: center;
    }
    h2 {
        font-size: 30px;
        color: #fff;
    }
    h3 {
        font-size: 24px;
        color: #fff;
    }
    p {
        font-size: 18px;
        color: #fff;
    }
    </style>
    """, unsafe_allow_html=True)


# Title of the app
st.title("Farming Challenges: Gamified NDVI Analysis and Agricultural Advice")

# User profile
st.sidebar.header("User Profile")
username = st.sidebar.text_input("Enter your username")
if not username:
    st.sidebar.warning("Please enter a username to start!")
    st.stop()

openweather_api_key = "8b7d1ed40b332b731beb9ee190eab6d9"
nasa_image_url = "https://lpdaac.usgs.gov/media/images/MOD13Q1_V6_28Jul2018_Brazil_Hero.original_dEF2JTV.jpg"

client = initialize_groq_client(groq_api_key)

# Game Variables
if 'points' not in st.session_state:
    st.session_state.points = 0
if 'quests_completed' not in st.session_state:
    st.session_state.quests_completed = []

# Select Location
location = st.sidebar.text_input("Enter Your Farm Location (e.g., 'New York')")
if not location:
    st.sidebar.warning("Please enter a location to get weather data!")
    st.stop()

# Fetch real data
precipitation, temperature = fetch_weather_data(openweather_api_key, location)

# Fetch NDVI image
ndvi_image = fetch_ndvi_image(nasa_image_url)


def calculate_area_in_m2(pixel_count, spatial_resolution):
    # spatial_resolution is in meters, e.g., 250 meters per pixel
    # Area of a pixel in square meters = spatial_resolution^2
    pixel_area_m2 = spatial_resolution ** 2
    total_area_m2 = pixel_count * pixel_area_m2
    return total_area_m2

def calculate_area_in_hectares(total_area_m2):
    # 1 hectare = 10,000 square meters
    total_area_hectares = total_area_m2 / 10_000
    return total_area_hectares

def calculate_area_in_sq_km(total_area_m2):
    # 1 sq km = 1,000,000 square meters
    total_area_km2 = total_area_m2 / 1_000_000
    return total_area_km2

# Display results in a more readable format
if ndvi_image:
    st.image(ndvi_image, caption="Fetched NDVI Image from NASA", use_column_width=True)

    processed_image = process_ndvi_image(ndvi_image)
    st.image(processed_image, caption="Processed NDVI Image", use_column_width=True)

    healthy_areas = np.sum(processed_image > 0)  # Count healthy areas
    total_area = processed_image.size
    health_percentage = (healthy_areas / total_area) * 100

    spatial_resolution = 250  # Example resolution

    # Calculate the area in square meters, hectares, and square kilometers
    healthy_area_m2 = calculate_area_in_m2(healthy_areas, spatial_resolution)
    healthy_area_hectares = calculate_area_in_hectares(healthy_area_m2)
    healthy_area_km2 = calculate_area_in_sq_km(healthy_area_m2)

    st.write(f"Healthy Vegetation Area: {healthy_area_m2:,.2f} square meters")
    st.write(f"Healthy Vegetation Area: {healthy_area_hectares:,.2f} hectares")
    st.write(f"Healthy Vegetation Area: {healthy_area_km2:,.2f} square kilometers")

    total_area_m2 = calculate_area_in_m2(total_area, spatial_resolution)
    st.write(f"Total Area: {total_area_m2:,.2f} square meters")
    st.write(f"Percentage of Healthy Vegetation: {health_percentage:.2f}%")

    
    # Update points for completing the NDVI quest
    if st.button("Complete NDVI Quest"):
        st.session_state.points += 10
        st.session_state.quests_completed.append("NDVI Analysis")
        st.sidebar.success("Quest completed: NDVI Analysis!")

    # Weather Analysis Section
    st.subheader("Weather Analysis")
    if precipitation is not None and temperature is not None:
        st.write(f"Precipitation in last hour: {precipitation} mm")
        st.write(f"Current Temperature: {temperature} °C")
        
        # Suggest crops based on weather conditions
        if precipitation > 0:
            st.write("Based on the recent precipitation, consider planting crops like rice or corn.")
        else:
            st.write("With no recent precipitation, drought-resistant crops like millet or sorghum may be more suitable.")
        
        if temperature < 15:
            st.write("The temperature is low; consider planting cold-resistant crops like kale or carrots.")
        elif 15 <= temperature <= 25:
            st.write("The temperature is ideal; you can plant a variety of crops including tomatoes and beans.")
        else:
            st.write("High temperatures suggest planting heat-resistant crops like cucumbers or peppers.")
        
        # LLM Query Interface
        st.subheader("Ask Agriculture Questions")
        query = st.text_area("Ask a question related to farming challenges or agricultural advice:")

        openweather_api_key = "8b7d1ed40b332b731beb9ee190eab6d9"
        precipitation, temperature = fetch_weather_data(openweather_api_key, location)
        context = f"The current precipitation is {precipitation} mm and the temperature is {temperature} °C. at {location}"
        user_query = f"{query} in context of the the conditions {context}."


        if user_query:
            if st.button("Get LLM Answer"):
                llm_answer = fetch_llm_answer(user_query, client)
                st.write(f"LLM Answer: {llm_answer}")
            
            if st.button("Get RAG-Enhanced Answer"):
                df, index, model, _= load_knowledge_base_and_embeddings()
                retrieved_docs = retrieve_documents(user_query, df, index, model)
                rag_answer = fetch_rag_answer(user_query, client, retrieved_docs)
                st.write(f"RAG-Enhanced Answer: {rag_answer}")
                
                llm_answer = fetch_llm_answer(user_query, client)
                final_response = fusion_response(llm_answer, rag_answer)
                st.write(f"Final Fusion Response:\n{final_response}")
