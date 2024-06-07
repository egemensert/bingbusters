import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import matplotlib.pyplot as plt

import pandas as pd
import pyterrier as pt


from deep_model_utils import DeepModel

if not pt.started():
    pt.init()

dataset = pt.datasets.get_dataset('irds:cord19/trec-covid')
# (Optional) Pre-process the dataset if feasible
doc_info_dict = {}
for doc in dataset.get_corpus_iter():
  doc_info_dict[doc['docno']] = {"title": doc.get("title", "No Title Available"), "abstract": doc.get("abstract", "")}

indexer = pt.index.IterDictIndexer('./cord19-index') # initialize an indexer object
indexref = indexer.index(dataset.get_corpus_iter(), fields=('title', 'abstract'))
index = pt.IndexFactory.of(indexref)
BM25_br = pt.BatchRetrieve(index, wmodel="BM25")

indexer_v2 = pt.index.IterDictIndexer('./cord19-index_v2',stemmer= None, stopwords = None) # initialize an indexer object
indexref_v2 = indexer_v2.index(dataset.get_corpus_iter(), fields=('title', 'abstract'))
index_v2 = pt.IndexFactory.of(indexref_v2)
BM25_br_vanilla = pt.BatchRetrieve(index_v2, wmodel="BM25")

model_dir = 'ceng596'
model_hub = {}
for model_name in ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2', 'all-distilroberta-v1']:
    m = DeepModel(model_path=f'{model_dir}/{model_name}', embeddings_path=f'{model_dir}/{model_name}.npy', documents_path=f'{model_dir}/documents.json')
    model_hub[model_name] = m

# Improved layout with clear separation and styling
app = dash.Dash(__name__)
app.layout = html.Div(
    style={"text-align": "center", "font-family": "Arial, sans-serif"},  # Center all content with a clean font
    children=[
        html.H1("BingBuster's Search Engine", style={"margin-bottom": "20px", "color": "#333"}),
        html.Div(
            style={"margin-bottom": "20px"},
            children=[
                html.Label("Enter your keyword:  ", style={"font-weight": "bold", "font-size": "16px"}),
                dcc.Input(id="keyword-input", type="text", value="", style={"width": "50%", "padding": "10px", "font-size": "16px", "border-radius": "10px" }),
                html.Label("Results per page:", style={"font-size": "14px", "margin-left": "20px", "margin-right": "10px"}),
                dcc.Input(id="result-limit-input", type="number", value=10, min=1, max=50, style={"width": "60px", "padding": "5px", "font-size": "14px"}),
            ],
        ),
        html.Div(
            children=[
                html.Label("Retrieval Model:", style={"font-size": "14px", "margin-right": "10px"}),
                dcc.RadioItems(
                    id="model-choice",
                    options=[
                        {"label": "BM25_br", "value": "BM25_br"},
                        {"label": "BM25_br_vanilla", "value": "BM25_br_vanilla"},
                        *[{"label": f'Vector similarity ({k})', "value": k} for k in model_hub.keys()]
                    ],
                    value="BM25_br",  # Default selection
                    labelStyle={"display": "inline-block", "margin-right": "10px", "font-size": "14px"},
                ),
            ],
            style={"margin-bottom": "20px"},
        ),
        html.Button(
            id="search-button",
            children="Search",
            n_clicks=0,
            style={
                "background-color": "#4CAF50",  # Green color
                "color": "white",
                "border": "none",  # Remove border
                "padding": "10px 20px",
                "font-size": "16px",
                "cursor": "pointer",
                "border-radius": "5px",  # Rounded corners
            },
        ),
        html.Div(id="search-results", children=[], style={"text-align": "left", "margin-top": "20px", "font-size": "16px"}),
    ],
)

# Callback with error handling and informative messages
@app.callback(
    Output(component_id="search-results", component_property="children"),
    [Input(component_id="search-button", component_property="n_clicks")],
    [
        Input(component_id="keyword-input", component_property="value"),
        Input(component_id="result-limit-input", component_property="value"),
        Input(component_id="model-choice", component_property="value"),  # New Input
    ],
)
def update_results(n_clicks, keyword, result_limit, selected_model):
    if n_clicks == 0 or not keyword:
        return []  # Handle initial state and empty input

    # Retrieve documents based on selected model
    try:
        if selected_model == "BM25_br":
            top_docs = retrieve_top_documents(keyword, BM25_br, dataset, result_limit)
        elif selected_model in model_hub:
            top_docs = model_hub[selected_model].retrieve_top_documents(keyword, result_limit)
        else:
            top_docs = retrieve_top_documents(keyword, BM25_br_vanilla, dataset, result_limit)

        if not top_docs:
            return "No relevant documents found."

        # Format results as HTML list
        results_list = html.Ol(
            children=[
                html.Li(children=f"{doc_info_dict[doc]['title']}", style={"margin-bottom": "5px"})
                for doc in top_docs[:result_limit]
            ]
        )
        return results_list
    except Exception as e:
        return f"Error: {str(e)}"  # Provide informative error message

# Function to retrieve top documents using PyTerrier (assuming you have a defined retrieval function)
def retrieve_top_documents(keyword, retrieval_model, dataset, result_limit):
    top_docs = retrieval_model.search(query=keyword).head(result_limit)  # Retrieve top based on result_limit
    # Extract a list of docids
    doc_ids = top_docs["docno"].tolist()
    return doc_ids  # Return the list of docids

if __name__ == "__main__":
    app.run_server(debug=True, host=8051)
