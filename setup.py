from setuptools import find_packages,setup

setup(
    name='PdfSage',
    version='0.0.1',
    author='Shivam Kumar',
    author_email='shivamkumarsingh1064@gmail.com',
    install_requires=["google-generativeai","langchain","streamlit","python-dotenv","PyPDF2","pinecone-client","langchain_google_genai","tiktoken"],
    packages=find_packages()
)