#!/usr/bin/env python3
from dotenv import load_dotenv
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Chroma
from langchain.llms import GPT4All, LlamaCpp
#~
from flask import Flask, request, jsonify
#~~
import os
import argparse
#~
app = Flask(__name__)
#~~

load_dotenv()

embeddings_model_name = os.environ.get("EMBEDDINGS_MODEL_NAME")
persist_directory = os.environ.get('PERSIST_DIRECTORY')

model_type = os.environ.get('MODEL_TYPE')
model_path = os.environ.get('MODEL_PATH')
model_n_ctx = os.environ.get('MODEL_N_CTX')
target_source_chunks = int(os.environ.get('TARGET_SOURCE_CHUNKS',4))

from constants import CHROMA_SETTINGS
#~

embeddings=0
db=0
retriever=0
callbacks = []
llm =0
qa=0

def main_start():
    # Parse the command line arguments
#    args = parse_arguments()

    global embeddings
    global db
    global retriever
    global callbacks
    global llm
    global qa
    
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model_name)
    db = Chroma(persist_directory=persist_directory, embedding_function=embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever(search_kwargs={"k": target_source_chunks})
    # activate/deactivate the streaming StdOut callback for LLMs
#    callbacks = [] if args.mute_stream else [StreamingStdOutCallbackHandler()]
    callbacks = []
    # Prepare the LLM
    match model_type:
        case "LlamaCpp":
            llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, callbacks=callbacks, verbose=False)
        case "GPT4All":
            llm = GPT4All(model=model_path, n_ctx=model_n_ctx, backend='gptj', callbacks=callbacks, verbose=False)
        case _default:
            print(f"Model {model_type} not supported!")
            exit;
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
"""
    # Interactive questions and answers
    while True:
        query = input("\nEnter a query: ")
        if query == "exit":
            break

        # Get the answer from the chain
        res = qa(query)
        answer, docs = res['result'], [] if args.hide_source else res['source_documents']

        # Print the result
        print("\n\n> Question:")
        print(query)
        print("\n> Answer:")
        print(answer)

        # Print the relevant sources used for the answer
        for document in docs:
            print("\n> " + document.metadata["source"] + ":")
            print(document.page_content)
"""

@app.route('/parivategpt', methods=['POST'])
def process_request():
    global qa
    
    data = request.get_json()  # Get the JSON data from the request
    query = data['query']  # Extract the input string

    # print(query)
    # Perform any processing or manipulation on the input string here
    res = qa(query)
    answer, docs = res['result'], res['source_documents']

    # Print the relevant sources used for the answer
    sources_doc=""
    for document in docs:
        sources_doc= sources_doc + "\n> " + document.metadata["source"] + ":"
        sources_doc= sources_doc + document.page_content


    # Prepare the response as JSON
    response = {
        'result': answer,
        'sources': sources_doc
    }

    return jsonify(response)



"""
def parse_arguments():
    parser = argparse.ArgumentParser(description='privateGPT: Ask questions to your documents without an internet connection, '
                                                 'using the power of LLMs.')
    parser.add_argument("--hide-source", "-S", action='store_true',
                        help='Use this flag to disable printing of source documents used for answers.')

    parser.add_argument("--mute-stream", "-M",
                        action='store_true',
                        help='Use this flag to disable the streaming StdOut callback for LLMs.')

    return parser.parse_args()
"""
#~~

if __name__ == "__main__":
#~
#    main()
    main_start()
    app.run(port=8888)
#~~
