# chat_with_pdf_st
Repo para validação de projeto 

## Dependencias e Instalação
----------------------------
Para instalar a aplicação, siga os seguintes passos:

1. Clone o repo para a sua máquina local.

2. Instale as dependências com o seguinte código:
   ```
   pip install -r requirements.txt
   ```

3. Insira a sua API key no arquivo `.env` em um diretório do projeto.
```commandline
OPENAI_API_KEY=your_secrit_api_key
```

## Utilização
-----
Para utilizar a aplicação siga os passos:

1. Garanta que você instalou todas as dependências do arquivo `requirements.txt` e criou o seu arquivo `.env`.

2. Rode o arquivo `app.py` com o comando Streamlit abaixo:
   ```
   streamlit run app.py
   ```
3. A aplicação estará disponível para ser acessada no seu navegador através do ip:8502

4. Carregue o documento PDF que desejar.

5. Realize perguntas para o modelo.
