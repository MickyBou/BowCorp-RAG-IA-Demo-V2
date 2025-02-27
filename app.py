
import streamlit as st
import pandas as pd
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
import os
from io import BytesIO, TextIOWrapper
import PyPDF2
import docx2txt
import csv
from huggingface_hub import InferenceClient

st.set_page_config("BowCorp RAG IA")
st.title(':crystal_ball: BowCorp RAG IA - Demo V2')
st.text('Escoge tu fuente/fuentes de conocimiento:')
st.text(' · Internet (Por defecto)\n · Una URL concreta\n · Un documento\n')
st.text('Escoge además opcionalmente un prompt especial.')

Model = "GEMINI"
tkey = st.secrets["GOOGLE_API_KEY"]

# Set up the model
generation_config = {
    "temperature": 0.9,
    "top_p": 1,
    "top_k": 1,
    "max_output_tokens": 2048,
}

safety_settings = [
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_MEDIUM_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_ONLY_HIGH",
    },
]

# model = genai.GenerativeModel(model_name="gemini-2.0-flash",
#                            generation_config=generation_config,
#                            safety_settings=safety_settings)

model = genai.GenerativeModel(model_name="gemini-2.0-flash-exp",
                            generation_config=generation_config,
                            safety_settings=safety_settings)

genai.configure(api_key=tkey)

def gai(inp):
    return model.generate_content(inp).text

# bg image
page_bg_img = """
<style>
[data-testid="stAppViewContainer"] {
background-image: url(
https://raw.githubusercontent.com/MickyBou/Playground/refs/heads/main/5968949.jpg
);
background-size: cover;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

inp = st.text_input("Inicia aquí tu conversación.", "")

# Function to scrape data
def scrape_data(url):
    # Send HTTP request and parse content
    response = requests.get(url)
    # print(response)
    soup = BeautifulSoup(response.content, 'html.parser')

    # Scraping logic - use BeautifulSoup to find and extract various types of content
    texts = [element.text for element in soup.find_all(['p', 'a', 'img'])]
    links = [element.get('href') for element in soup.find_all('a') if element.get('href')]
    images = [element.get('src') for element in soup.find_all('img') if element.get('src')]

    # Ensure all lists are of the same length by padding the shorter ones with None
    max_length = max(len(texts), len(links), len(images))
    texts += [None] * (max_length - len(texts))
    links += [None] * (max_length - len(links))
    images += [None] * (max_length - len(images))

    # Create a DataFrame using pandas for texts, links, and images
    data = {'Text': texts, 'Links': links, 'Images': images}
    df = pd.DataFrame(data)

    # return the processed data
    return df

# Function to extract text from a PDF file
def extract_text_from_pdf(file_bytes):
    pdf_reader = PyPDF2.PdfReader(BytesIO(file_bytes))
    num_pages = len(pdf_reader.pages)

    text = ""
    for page_num in range(num_pages):
        page = pdf_reader.pages[page_num] 
        text += page.extract_text() 

    return text.replace('\t', ' ').replace('\n', ' ')

# Function to extract text from a TXT file
def extract_text_from_txt(file_bytes):
    text = file_bytes.decode('utf-8')
    return text

# Function to extract text from a DOCX file
def extract_text_from_docx(file_bytes):
    docx = docx2txt.process(BytesIO(file_bytes))
    return docx.replace('\t', ' ').replace('\n', ' ')

def extract_text_from_csv(file_bytes, encoding='utf-8'):
    # Convert bytes to text using the specified encoding
    file_text = file_bytes.decode(encoding)

    # Use CSV reader to read the content
    csv_reader = csv.reader(TextIOWrapper(BytesIO(file_text.encode(encoding)), encoding=encoding))
    
    # Concatenate all rows and columns into a single text
    text = ""
    for row in csv_reader:
        text += ' '.join(row) + ' '

    return text.replace('\t', ' ').replace('\n', ' ')

url_input = st.checkbox("Utilizar una  URL concreta")
url = ""
if url_input:
    # Input for the website URL
    url = st.text_input('Introduce la URL: ', '')

file_input = st.checkbox("Utilizar fichero")
uploaded_file = None

if file_input:
    # Add file uploader
    st.write("Sube un fichero PDF, TXT, o DOCX para extraer el texto.")
    uploaded_file = st.file_uploader("Escoge un fichero")

    if uploaded_file:
    # Get the file extension
        file_name, file_extension = os.path.splitext(uploaded_file.name)

        if file_extension:
            # Extract text based on the file extension
            if file_extension == ".pdf":
                uploaded_file = extract_text_from_pdf(uploaded_file.getvalue())
            elif file_extension == ".txt":
                uploaded_file = extract_text_from_txt(uploaded_file.getvalue())
            elif file_extension == ".docx":
                uploaded_file = extract_text_from_docx(uploaded_file.getvalue())
            elif file_extension == ".csv":
                uploaded_file = extract_text_from_csv(uploaded_file.getvalue())

            else:
                st.error("Formato de fichero erróneo.")

sp_prompt = ""
prompt_input = st.checkbox("Usar indicación especial")
if prompt_input:
    sp_prompt = st.selectbox("Indicación especial (opcional):", [
        "Prompt A: Intenta resumir tu respuesta en 5 líneas.",
        "Prompt B: Responde de la forma más extensa posible.",
        "Prompt C: Por favor, responde en Catalán."
    ])

output = ''
previous_responses = []
if st.button("Generate"):
    if tkey == '':
        st.error("Error con la clave API.")

    if url:
        if 'https://' not in url:
            url = 'https://' + url
        scraped_data = scrape_data(url)
        paragraph = ' '.join(scraped_data['Text'].dropna())
        # st.write(scraped_data)
        # st.write(paragraph)

        inp = paragraph + ' ' +"Take the given data above, as information and generate a response based on this prompt: " + inp       

    if sp_prompt:
        inp = inp + " " + sp_prompt
    if uploaded_file:
        inp = inp + " " + uploaded_file

    if inp:
        # st.write(inp)
        output = gai(inp)
        st.write(output)

        # # Add response to the list of previous_responses
        # previous_responses.append(output)

        # # Display all previous responses
        # st.subheader("Previous Responses:")
        # for i, response in enumerate(previous_responses, start=1):
        #     st.write(f"{i}. {response}")


        # Add download button
        if output is not None:
            # filename = 'Generated_Answer.txt'
            # with open(filename, 'w') as f:
            #     f.write(output)

            # Add select box
            ofType = 'txt'
            #ofType = st.selectbox("Chose an output file type: ", ["TXT", "PY", "HTML"])
            st.download_button("Descargar fichero", data = output, file_name= f"Generated Answer.{ofType}")
    else:
        st.error("Por favor entra una consulta para generar una respuesta.")

#st.subheader("[🔗...Visit my GitHub Profile...🔗](https://github.com/NafisRayan)")

# streamlit run app.py
