import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
#from langchain.llms import HuggingFaceHub
#from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    #embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    #llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    # teacher_like_prompt = r"""
    # Você é um auxiliar de manutenção.
    # Forneca respostas que auxiliem o manutentor na realização dos reparos.
    # ----
    # {context}
    # ----
    # """

    # Create a ChatPromptTemplate with the teacher-like prompt
    #teacher_prompt = ChatPromptTemplate.from_template(teacher_like_prompt)

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory,
        #combine_docs_chain_kwargs={'prompt': teacher_prompt}
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Converse com o seus PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Converse com o seu PDF :books:")
    st.caption('É necessário que o processamento seja feito ao menos 1 vez antes de realizar a pergunta. \
                 Isso pode ser feito apertando o botão "Processar"')
    
    user_question = st.text_input("Faça uma pergunta para o documento que você inseriu:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Seus Documentos")
        pdf_docs = st.file_uploader(
            label='Faça o upload dos seus arquivos PDF e clique no botão "Processar" ', 
            accept_multiple_files=True, 
            help="Arraste e solte seus arquivos PDF aqui")
        if st.button("Processar"):
            with st.spinner("Processando"):
                # Recebe o texto do PDF
                raw_text = get_pdf_text(pdf_docs)

                # Recebe o texto em pedaços
                text_chunks = get_text_chunks(raw_text)

                # Cria a "vector store"
                vectorstore = get_vectorstore(text_chunks)

                st.session_state.conversation = get_conversation_chain(vectorstore)
        
        if st.button("Habilitar nova conversa"):
            # Cria a sequência de conversa
            st.session_state.conversation.memory.clear()

    if isinstance(st.session_state.conversation, ConversationalRetrievalChain):
        if st.session_state.conversation.memory.chat_memory.messages:
            feedback = st.text_area("""A resposta não lhe atendeu? 
                                    Dê um feedback sobre a resposta retornada, a nossa equipe analisará o caso.""")
            if feedback:
                # Aqui você pode processar o feedback, por exemplo, enviando por e-mail ou armazenando em um banco de dados
                pass
if __name__ == '__main__':
    main()
