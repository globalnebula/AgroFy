import streamlit as st
import pandas as pd
import requests
import numpy as np
import cv2
from PIL import Image
import io
from groq import Groq


groq_api_key = "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" 

# Initialize Groq client
def initialize_groq_client(api_key):
    client = Groq(api_key=api_key)
    return client

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

# Function to fetch NDVI image from NASA
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
                "content": "You are an AI assistant designed to support farmers by providing accurate and timely information related to agriculture. You understand various aspects of farming, including crop management, soil health, weather forecasting, pest control, irrigation techniques, market prices, government schemes, and modern farming technologies."
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

# Title of the app
st.title("Farming Challenges: Gamified NDVI Analysis")

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

if ndvi_image:
    st.image(ndvi_image, caption="Fetched NDVI Image from NASA", use_column_width=True)

    # Process NDVI image
    processed_image = process_ndvi_image(ndvi_image)
    
    # Display processed image
    st.image(processed_image, caption="Processed NDVI Image", use_column_width=True)
    
    # Perform agricultural analysis
    healthy_areas = np.sum(processed_image > 0)  # Count healthy areas
    total_area = processed_image.size
    health_percentage = (healthy_areas / total_area) * 100

    st.write(f"Healthy Vegetation Area: {healthy_areas} pixels")
    st.write(f"Total Area: {total_area} pixels")
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
    else:
        st.error("Failed to fetch weather data.")

else:
    st.error("Failed to fetch NDVI image.")

# Sidebar for showing points and completed quests
st.sidebar.write(f"Points: {st.session_state.points}")
st.sidebar.write("Quests Completed:")
for quest in st.session_state.quests_completed:
    st.sidebar.write(f"- {quest}")

# Feedback Section
st.subheader("Feedback on Your Analysis")
feedback = st.text_area("Enter your feedback on the data:")
if st.button("Submit Feedback"):
    st.success("Feedback submitted! Thank you for your insights.")

# Agriculture Chatbot Section
st.subheader("Ask an Agriculture Expert")
query = st.text_input("Enter your agriculture-related query:")
if st.button("Get Advice"):
    if query:
        advice = fetch_agriculture_advice(query, client, location)
        st.write(f"Expert Advice: {advice}")
    else:
        st.warning("Please enter a query.")

# Achievement Badges
st.subheader("Achievements")
if health_percentage >= 75:
    st.write("🏆 Achievement Unlocked: Green Thumb! 🎉")