def read_and_textify(files):
    text_list = []
    source_list = []
    for file in files:
        pdfreader = PyPDF2.PdfReader(file)
        for i in range(len(pdfreader.pages)):
            pageobj = pdfreader.getPage[i]
            text = pageobj.extract_text()
            pageobj.clear()
            text_list.append(text)
            sources_list.append(file.name + "_page_" + str(i))
    return [text_list, sources_list]

vectordb = Chroma.from_texts(documents, embeddings, metadatas = [{"source":s} for s in sources] persist_directory=persist_directory) 
vectordb.persist()
vectordb = None
vectordb = Chroma(persist_directory=persist_directory,embeddings_function=embeddings)
vectordb.get()       