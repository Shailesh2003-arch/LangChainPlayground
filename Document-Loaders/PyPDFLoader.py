from langchain_community.document_loaders import PyPDFLoader



document_loader = PyPDFLoader('./Node JS.pdf')

loaded_documents = document_loader.load()

print(loaded_documents)

# PYPDFLoader doesnt work great with scanned or layout documents.

# For more visit or use these...

# 1) PDFs with tables/columns - PDFPlumberLoader
# 2) Scanned / Images PDFs -  UnstructuredPDFLoader or AmazonTextractPDF
# 3) Need layout and Image data - PyMuPDFLoader
# 4) Want best structure extraction - UnstructuredPDFLoader

# There are also different types of loaders like CSVLoader, DirectoryLoader - which loads PDFs, CSVs and different format of files into the loader.
# For more check official documentation.