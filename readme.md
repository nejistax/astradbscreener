# Resume Screener with AstraDB & LangChain.

## PROJECT OVERVIEW

This application demonstrates the use of ASTRADB as a vector store for matching a job description against a number of resumes. The hiring manager provides a job description and K matching results, then uploads multiple resumes and the application reviews and matches the best fit for the job position. The application also generates a summary for each resume and tries to justify why the particular resume was selected as a good match for the job position.
The application is written in Python. Streamlit is used for the front end.

To run the application locally:
1. Pull the repo using Git
2. Open the project in an IDE of your choice
3. Create a .env file and add the following environent variables: ASTRA_DB_APPLICATION_TOKEN, ASTRA_DB_API_ENDPOINT, ASTRA_DB_NAMESPACE,ASTRA_DB_COLLECTION, OPENAI_API_KEY




### THINGS YOU WILL NEED

- An AstraDB vector database 

- An Open AI key - https://www.openai.com

- Streamlit for the UI 


### CREATE A DATABASE ON ASTRADB

Sign up for an AstraDB account and create a database. You can follow the instructions here: https://www.datastax.com/. You can name this database anything you want, just take note of your application token and API endpoint. 


### GET AN OPENAI Key

Visit https://openai.com and sign up for an API key. 

### INSTALL STREAMLIT LOCALLY

Streamlit lets you build beautiful UIs in record time using just Python. You can learn more here: https://streamlit.io/.  Run the following command  - pip install streamlit 


### LET’S GET STARTED

Create a new python project using the iDE of your choice. Inside this project, create a .env file and a requirements.txt file for our dependencies.

Inside the .env create the following environment variables and assign them to their values you 

Open requirements.txt and add the following:
-  astrapy
-  openai
-  langchain
-  streamlit
-  tiktoken
-  pypdf
-  python_dotenv
-  unstructured

  
Run pip install -r requirements.txt

Create a python file called utils.py. This file shall contain some of our utility functions. 

**INITIALIZE ASTRADB**

Inside utils.py, add the following:

```from langchain.chains.summarize import load_summarize_chain
  from astrapy.db import AstraDB
  import os
  from langchain.chat_models import ChatOpenAI
  from dotenv import find_dotenv, load_dotenv
  load_dotenv(find_dotenv(), override=True)
```

Create the following function to initialize AstraDB:

```def init_astra_db():
   db = AstraDB(
       namespace=os.environ["ASTRA_DB_NAMESPACE"],
       token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
       api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"])
  print(f"Connected to Astra DB: {db.get_collections()}")
```
  

Still inside the utils.py, we are going to create the following utility functions:

Load Document: This loads our resumes and supports multiple formats like pdfs, docx, etc: We use the PyPDFLoader from LangChain.
```

def load_any_document(file):
   name, extension = os.path.splitext(file)

   if extension == ".pdf":
       from langchain.document_loaders import PyPDFLoader


       print(f"loading file {file}")
       loader = PyPDFLoader(file)


   elif extension == ".docx":
       from langchain.document_loaders import Docx2txtLoader


       print(f"loading file {file}")
       loader = Docx2txtLoader(file)
   elif extension == ".txt":
       from langchain.document_loaders import TextLoader


       print(f"loading file {file}")
       loader = TextLoader(file)


   else:
       print("Non supported document format")
       return None


   data = loader.load()
```

## DOCUMENT CHUNKING

We are going to split our documents into chunks using the following function. We set a default chunk size of 256 with an overlap of 10. You can try different values to see what works best . We then use the RecursiveCharacterTextSplitter from LangChain to split our text into these chunks and return LangChain Documents.
```
def chunk_data(doc, chunk_size=256, chunk_overlap=10, metadata={}):
   from langchain.text_splitter import RecursiveCharacterTextSplitter


   text_splitter = RecursiveCharacterTextSplitter(
       chunk_size=chunk_size, chunk_overlap=chunk_overlap, separators=["\n", " \n"]
   )


   docs = text_splitter.split_documents(doc)
   for doc in docs:
       doc.metadata = metadata


   return docs
```

## UPLOAD FILES

Inside utils.py, our create_docs functions lets up upload the resumes and chunk them into LangChain Documents by calling the chunk_data function. We specify a chunk size of 1000 with an overlap of 200. 
```
def create_docs(user_pdf_list, unique_id):
   output_docs = []
   file_names = []
   for document in user_pdf_list:
       bytes_data = document.read()
       file_names.append(document.name)


       file_name = os.path.join("./uploads", document.name)
       with open(file_name, "wb") as f:
           f.write(bytes_data)
       docs = load_any_document(file_name)


       chunks = chunk_data(doc=docs,
           chunk_size = 1000,
           chunk_overlap = 200,
           metadata={"name": document.name,
                     "type": document.type,
                     "size": document.size,
                     "unique_id": unique_id},)
       output_docs.extend(chunks)

   return output_docs
```

### SUMMARIZE EACH RESUME

We create a function that summarizes an uploaded resume load_summarize_chain from LangChain. We pass in an LLM instance of ChatOpenAI with chain_type of map reduce.
```
def get_summary(current_doc):
   llm = ChatOpenAI()
   chain = load_summarize_chain(llm=llm, chain_type="map_reduce")
   summary = chain.run([current_doc])
   return summary
```

### THE FRONT END

Inside main.py, add the following imports:
```
import os
from langchain.embeddings import OpenAIEmbeddings
import streamlit as st
import uuid
from utils import *
from dotenv import find_dotenv, load_dotenv
from langchain.vectorstores import AstraDB
```


Then create the following function:
```
def main():
   load_dotenv(find_dotenv(), override=True)
   st.set_page_config(page_title="AstraDB Resume Screening Assistant")
   st.title('AstraDB Resume Screening Assistant')
   st.subheader('I can help you screen candidates resumes')
   load_dotenv(find_dotenv(), override=True)


   init_astra_db()


   job_description = st.text_area("Please post your 'JOB DESCRIPTION' here..", key=1)
   document_count = st.text_input("No. of matching RESUMES to return ", key=2)


   # upload the resumes
   pdfs = st.file_uploader("Upload resumes here. Only PDF files are allowed", type=["pdf"], accept_multiple_files=True)
   submit = st.button('Analyze resumes')


   if submit:
       with st.spinner('Uploading and analyzing documents'):


           st.session_state["unique_id"] = uuid.uuid4().hex
           # create the documents list from all the uploaded pdfs
           docs = create_docs(pdfs, st.session_state["unique_id"])
           # display the count of resumes that have been uploaded
           st.write(len(docs))


           # create emebeddings instance
           embedding = OpenAIEmbeddings()
           vstore = AstraDB(
               namespace=os.environ["ASTRA_DB_NAMESPACE"],
               embedding=embedding,
               collection_name=os.environ["ASTRA_DB_COLLECTION"],
               token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
               api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
           )


           inserted_docs = vstore.add_documents(docs)
           st.write(f"\nInserted {len(inserted_docs)} documents into vector store.")


           #now perform similarity search to show candidates
           relevant_docs = (vstore.similarity_search(job_description, k=document_count))


           for item in range(len(relevant_docs)):
               st.subheader("👉" + str(item + 1))
               # display the filepath
               st.write("**File** : " + relevant_docs[item].metadata["name"])
               # add expander
               with st.expander("Display Summary 👀"):
                   summary = get_summary(relevant_docs[item])
                   st.write(summary)
           st.success('Hope I was able to save your time... Thanks')




if __name__ == '__main__':
   main()
```
### Run the application
Run the following command - streamlit run main.py

