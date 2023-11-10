Streamlit app 📄🤖💬leveraging LaMini-T5-738M for PDF summarization and interactive querying, 🚀🔗🧩integrating blockchain for transaction tracking and encouraging contributions. 

📁Block.py 
It consists of a Block class representing individual blocks in the blockchain. Each block contains an index, a list of transactions, the previous block's hash, a nonce, and a timestamp. The generate_hash method creates a hash for the block's data. The add_transaction method appends new transactions to the block. Finally, the code demonstrates creating a block, adding a transaction, and displaying block information. This is a basic representation of how a 📁🔐💼 blockchain structure works for storing transaction data securely.

📁Blockchain.py
 It includes a Blockchain class with methods to create a blockchain, add blocks with proof-of-work mining, and validate the chain's integrity📊📈🔍. The mining method adds new blocks by performing proof of work and clearing pending transactions. The proof_of_work method calculates the hash with a specific difficulty. The check_chain_validity method checks the validity of the entire chain. It's a basic representation of a blockchain system with mining and transaction handling for secure data storage.
 
 📁Constants.py
 In this code snippet, we are configuring and initializing a database system known as "ChromaDB"📂📈🔗. This system allows us to manage and store data efficiently. The code imports essential modules, such as os for interacting with the operating system, and the chromadb library, which is presumably a Python package that provides functionality for working with ChromaDB.
Furthermore, we set specific configuration options using a Settings object, including the directory where data will be stored and whether anonymized telemetry is enabled. This configuration helps tailor ChromaDB to our needs, ensuring that it functions optimally for our data management requirements.

📁Summary.py 
Users can upload PDF files for summarization📄👥💡, which are then processed through a language model (LLM). The code includes functions to load and preprocess PDF files, as well as to display the uploaded PDFs and their corresponding summaries. It uses the 'transformers' library to work with a specific pre-trained language model called "MBZUAI/LaMini-T5-738M" for text summarization. The application's web interface allows users to upload PDF files, summarize them, and view the resulting summaries alongside the original documents.

📁ingest.py 
It uses various components from the 'langchain' library to load documents, such as PDF files, from a specified directory.
The documents are split into smaller text chunks using a 'RecursiveCharacterTextSplitter,' which allows for more efficient handling of large documents.
The script employs Sentence Transformer embeddings to convert the text chunks into numerical representations suitable for searching and retrieval.
A 'Chroma' vector store is created to store these embeddings, allowing for efficient and scalable document querying.
The ingestion process is completed by persisting the vector store, making it ready for document queries using another script called 'privateGPT.py.' This code streamlines the document ingestion and indexing process, making it easier to search and retrieve information from the ingested documents.

📁req.py 
This Python script aims to upgrade packages listed in a requirements file. It reads the requirements file, attempts to upgrade each package using the 'pip' module, and records successfully upgraded packages. If a package encounters dependency conflicts, it will be added to the conflicting_packages set, and the script will retry installing these conflicting packages to ensure they are upgraded. The primary goal is to update packages while handling dependency issues, providing an easier way to manage package updates in a Python project.

📁downgrade.py 
This Python script automates the process of uninstalling and reinstalling libraries📦🔄🛠️ specified in a 'req.txt' file. It first reads the 'req.txt' file to identify libraries for uninstallation and then attempts to uninstall them using 'pip'. Any uninstallation errors are recorded in the 'uninstall_errors' list.
Next, the script tries to install the libraries listed in 'req.txt' using 'pip'. If there are any installation errors, they are recorded in the 'install_errors' list.
Finally, the script prints any uninstallation errors, followed by any installation errors, providing a summary of the libraries that encountered issues during the process. This code streamlines the library management process by handling uninstallation and reinstallation of packages specified in 'req.txt'.

📁Chatbot_app.py 
chatbot_app.py is a Python script that implements a web-based application for interacting with a chatbot capable of summarizing PDF documents and providing responses to user queries. The script combines several key features and libraries to offer a comprehensive user experience:
Library Integration: It imports multiple Python libraries and dependencies, including Streamlit, Transformers, and Torch. Additionally, it utilizes custom libraries and modules, such as Blockchain, Block, and components from the langchain library.
Initialization and Configuration: The script initializes the Streamlit application and sets the device configuration to use the CPU. It configures the Hugging Face Transformers library by loading a pre-trained language model (T5) and tokenizer. It also establishes a blockchain system to securely store transaction data.
Data Ingestion and Storage: The script defines a function for 📄👥💡data ingestion, allowing it to process and store PDF documents. The ingestion process involves breaking down documents into text chunks using the langchain library, creating embeddings through Sentence Transformers, and storing the data in a Chroma vector store for efficient querying.
PDF Display and Summarization📝👤💬: The application provides functions for displaying uploaded PDF files, generating summaries for these documents, and recording the summaries in the blockchain. Users can upload PDFs, view their contents, and access concise document summaries.
Conversation Interface: Users can engage with the chatbot by inputting questions or instructions via a text input field. After submitting queries, the chatbot processes the input, utilizes the T5 language model to generate answers, and displays a history of the conversation.
User Interface: The Streamlit-based user interface features clear instructions, file upload capabilities, a chat input area, and a conversation history display, making it user-friendly and accessible.
Caching and State Management: The script optimizes performance by utilizing Streamlit's caching feature for specific functions, such as data ingestion and language model setup. It also manages the conversation history and past responses, enabling users to maintain a continuous conversation with the chatbot.
Main Function: The main function serves as the application's core, orchestrating the entire user experience. It structures the UI components, handles user interactions, and processes chatbot responses.
Execution Entry Point: The if __name__ == "__main__" block ensures that the script's main function is executed when the file is run as a standalone application.
In essence, chatbot_app.py is a robust application that harnesses the capabilities of pre-trained language models, blockchain technology, and various libraries to offer users a versatile platform for summarizing PDF documents, engaging in interactive conversations with a chatbot, and securely storing transaction data. It streamlines document management, enhances information retrieval, and fosters user interaction through an intuitive web interface.

The main code script enables users to upload PDF files for summarization, leveraging the "MBZUAI/LaMini-T5-738M" language model and Streamlit for the web interface. It also integrates with a blockchain to store summaries securely.
Block.py defines a Block class for the blockchain, handling transactions, block creation, and hashing. It demonstrates creating a block, adding a transaction, and displaying block information.
Blockchain.py contains the Blockchain class, responsible for managing the blockchain's creation, adding new blocks through mining, and validating the chain's integrity. It's a fundamental implementation of a blockchain structure.
Constants.py configures and initializes ChromaDB, a database system. It sets options for data storage directories and telemetry settings, tailoring ChromaDB for efficient data management.
Summary.py offers a user-friendly interface for summarizing uploaded PDF files using the "MBZUAI/LaMini-T5-738M" model. Users can view the original PDFs and their summaries via Streamlit.
Ingest.py automates document ingestion and indexing, utilizing the 'langchain' library. It processes PDF files, splits them into text chunks, generates embeddings, and stores them in a vector store. This code simplifies the process of making large volumes of text data searchable and retrievable.📁📈🌐
