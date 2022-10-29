import streamlit as st
import arxiv
from datetime import datetime, timedelta
from sentence_transformers import SentenceTransformer
import pytz
import torch
from operator import itemgetter
import numpy as np

utc=pytz.UTC

def find_papers(model, limit_date):
    out = []
    search = arxiv.Search(
        query = "cat:stat.ML OR cat:cs.AI",
        sort_by = arxiv.SortCriterion.SubmittedDate
    )
    infos = []
    for result in search.results():
        if result.published.replace(tzinfo=utc) <= limit_date:
            break
        infos.append((result.title, result.summary,result.pdf_url ))

    summaries = list(map(itemgetter(1), infos))
    emb = model.encode(summaries)
    for i in range(len(infos)):
        out.append((infos[i][0], emb[i], infos[i][2]))
    return out

def get_topk(out, text_emb, k=10):
    embeddings = list(map(itemgetter(1), out))
    titles = list(map(itemgetter(0), out))
    links = list(map(itemgetter(2), out))
    embeddings = torch.Tensor(embeddings)
    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    output = cos(embeddings, text_emb)
    v, i = torch.topk(output,k=k)
    indices = i.detach().numpy()
    for item in indices:
        st.markdown(f"[{titles[item]}]({links[item]}) Score: {str(np.around(output[item].detach().numpy(), 2))}")
        

model_id = "sentence-transformers/all-MiniLM-L6-v2"
st.header("Find Arxiv papers")
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
text_ref = st.text_area("Input text (title, abstract)")
date_ref = st.slider("Max n days ago", 1, 365)
topk = st.slider("Get top k result", 5, 30)
btn_run = st.button("Find papers")

ref_time = (datetime.now() - timedelta(days=date_ref)).replace(tzinfo=utc)
if btn_run:
    text_emb = torch.from_numpy(model.encode(text_ref))
    out = find_papers(model, ref_time)
    get_topk(out, text_emb, topk)
    
    


