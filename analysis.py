"""Graph construction, interpretable graph features and compact PyTorch GNN implementations."""
from __future__ import annotations
from dataclasses import dataclass
import math, re, time
import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch import nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score, confusion_matrix)
from sklearn.model_selection import train_test_split

torch.manual_seed(17)
np.random.seed(17)

ALIASES = {
    "source_user": ["source_user","src_user","author_id","user_id","from_user","source","sender_id"],
    "target_user": ["target_user","dst_user","reply_to_user_id","mentioned_user_id","to_user","target","recipient_id"],
    "text": ["text","comment","tweet","content","message"],
    "is_abusive": ["is_abusive","abusive","harassment","label","oh_label","cyberbullying"],
    "timestamp": ["timestamp","created_at","date","time"],
    "interaction_type": ["interaction_type","type","action"],
    "user_label": ["user_label","source_user_label","hateful","user_class"]
}

def infer_columns(df: pd.DataFrame):
    lower = {str(c).strip().lower(): c for c in df.columns}
    return {k: next((lower[a] for a in vals if a in lower), None) for k, vals in ALIASES.items()}

def _to_binary(series):
    if series is None: return None
    if pd.api.types.is_numeric_dtype(series):
        vals = pd.to_numeric(series, errors="coerce")
        return vals.map(lambda x: int(x > 0) if pd.notna(x) else np.nan)
    mapping = {"1":1,"true":1,"yes":1,"abusive":1,"hate":1,"hateful":1,"toxic":1,
               "harassment":1,"cyberbullying":1,"positive":1,"0":0,"false":0,"no":0,
               "normal":0,"non-abusive":0,"non_abusive":0,"clean":0,"negative":0}
    return series.astype(str).str.strip().str.lower().map(mapping)

def validate_interactions(df: pd.DataFrame, mapping=None):
    if df is None or df.empty: raise ValueError("The uploaded CSV has no rows.")
    mapping = mapping or infer_columns(df)
    src, dst = mapping.get("source_user"), mapping.get("target_user")
    if not src or not dst: raise ValueError("A user-interaction graph requires source_user and target_user columns. This file appears to be content-only or needs column mapping.")
    if src not in df or dst not in df: raise ValueError("Mapped source/target columns do not exist.")
    out = df.copy()
                                                                                 
                                                                                
    valid_ids = out[src].notna() & out[dst].notna()
    out = out.loc[valid_ids].copy()
    out["source_user"] = out[src].astype(str).str.strip()
    out["target_user"] = out[dst].astype(str).str.strip()
    invalid_tokens = {"", "nan", "none", "null", "<na>"}
    out = out[
        ~out.source_user.str.lower().isin(invalid_tokens)
        & ~out.target_user.str.lower().isin(invalid_tokens)
    ]
    if out.empty: raise ValueError("No valid source-to-target interactions remain after removing missing identifiers.")
    text_col = mapping.get("text")
    out["text"] = out[text_col].fillna("").astype(str) if text_col and text_col in out else ""
    abuse_col = mapping.get("is_abusive")
    out["is_abusive"] = _to_binary(out[abuse_col]) if abuse_col and abuse_col in out else np.nan
    ts_col = mapping.get("timestamp")
    out["timestamp"] = pd.to_datetime(out[ts_col], errors="coerce", utc=True) if ts_col and ts_col in out else pd.NaT
    type_col = mapping.get("interaction_type")
    out["interaction_type"] = out[type_col].fillna("interaction").astype(str) if type_col and type_col in out else "interaction"
    report_col = mapping.get("report_count")
    out["report_count"] = pd.to_numeric(out[report_col], errors="coerce").fillna(0) if report_col and report_col in out else 0.0
    label_col = mapping.get("user_label")
    if label_col and label_col in out:
        out["user_label"] = _to_binary(out[label_col])
    elif "user_label" in out.columns:
        out["user_label"] = _to_binary(out["user_label"])
    else:
        out["user_label"] = np.nan
    out["is_abusive"] = out["is_abusive"].clip(0,1)
    return out.reset_index(drop=True)

def build_graph(interactions: pd.DataFrame):
    G = nx.DiGraph()
    for row in interactions.itertuples():
        u, v = row.source_user, row.target_user
        if G.has_edge(u,v):
            G[u][v]["weight"] += 1
            if pd.notna(row.is_abusive): G[u][v]["abusive_count"] += int(row.is_abusive)
            G[u][v]["texts"].append(row.text[:180] if row.text else "")
            if pd.notna(row.timestamp):
                old = G[u][v].get("timestamps", [])
                old.append(row.timestamp)
                G[u][v]["timestamps"] = old
        else:
            G.add_edge(u,v,weight=1,abusive_count=int(row.is_abusive) if pd.notna(row.is_abusive) else 0,
                       labelled_count=int(pd.notna(row.is_abusive)), texts=[row.text[:180] if row.text else ""],
                       timestamps=[row.timestamp] if pd.notna(row.timestamp) else [])
                                                                                       
    nodes = sorted(G.nodes())
    return G, nodes

def node_features(G: nx.DiGraph, interactions: pd.DataFrame, nodes):
    und = G.to_undirected()
    pr = nx.pagerank(G, weight="weight") if len(nodes) else {}
                                                                                
                                                                       
    bet = nx.betweenness_centrality(und, k=min(100, len(nodes)), seed=17, weight=None) if len(nodes)>2 else {n:0 for n in nodes}
    clustering = nx.clustering(und, weight="weight")
                                                                             
    report_by_user = {}
    if "report_count" in interactions:
        report_by_user = interactions.groupby("source_user")["report_count"].sum().add(
            interactions.groupby("target_user")["report_count"].sum(), fill_value=0
        ).to_dict()
    rows=[]
    user_labels = {}
    if "user_label" in interactions:
        for u, part in interactions.groupby("source_user"):
            vals=part.user_label.dropna()
            if len(vals): user_labels[u]=int(vals.mode().iloc[0])
    for u in nodes:
        incoming=[d for _,_,d in G.in_edges(u,data=True)]
        outgoing=[d for _,_,d in G.out_edges(u,data=True)]
        inc_total=sum(d["weight"] for d in incoming); out_total=sum(d["weight"] for d in outgoing)
        inc_ab=sum(d["abusive_count"] for d in incoming)
        out_ab=sum(d["abusive_count"] for d in outgoing)
        abuse_received=inc_ab/max(1,sum(d["labelled_count"] for d in incoming))
        abuse_sent=out_ab/max(1,sum(d["labelled_count"] for d in outgoing))
        reports=float(report_by_user.get(u, 0.0))
        rows.append({
            "user_id":u, "in_degree":G.in_degree(u), "out_degree":G.out_degree(u),
            "interaction_volume":inc_total+out_total, "abuse_received_rate":abuse_received,
            "abuse_sent_rate":abuse_sent, "pagerank":pr.get(u,0),
            "betweenness":bet.get(u,0), "clustering":clustering.get(u,0),
            "reports":reports, "label":user_labels.get(u,np.nan)
        })
    table=pd.DataFrame(rows).set_index("user_id")
    feature_cols=["in_degree","out_degree","interaction_volume","abuse_received_rate",
                  "abuse_sent_rate","pagerank","betweenness","clustering","reports"]
    X=table[feature_cols].replace([np.inf,-np.inf],0).fillna(0).astype(float).to_numpy()
    for j in [0,1,2,8]: X[:,j]=np.log1p(X[:,j])
    return table, X, feature_cols

def community_analysis(G):
    U=G.to_undirected()
    if len(U)==0: return {}
    try:
        communities=nx.community.louvain_communities(U,weight="weight",seed=17)
    except Exception:
        communities=list(nx.community.greedy_modularity_communities(U,weight="weight"))
    return {n:i+1 for i,com in enumerate(communities) for n in com}

def repeated_targeting(interactions, min_sources=3, min_abuse_rate=0.4):
    labelled=interactions.dropna(subset=["is_abusive"])
    abusive=labelled[labelled.is_abusive==1]
    if abusive.empty: return []
    agg=abusive.groupby("target_user").agg(
        abusive_interactions=("source_user","size"),
        distinct_sources=("source_user","nunique"),
        first_seen=("timestamp","min"), last_seen=("timestamp","max")
    ).reset_index()
    total=labelled.groupby("target_user").size().rename("all_interactions")
    agg=agg.join(total,on="target_user")
    agg["abuse_rate"]=agg.abusive_interactions/agg.all_interactions.clip(lower=1)
    agg=agg[(agg.distinct_sources>=min_sources)&(agg.abuse_rate>=min_abuse_rate)]
    return agg.sort_values(["distinct_sources","abusive_interactions"],ascending=False).to_dict("records")

def temporal_bursts(interactions, window="1h"):
    d=interactions.dropna(subset=["timestamp","is_abusive"])
    d=d[d.is_abusive==1]
    if d.empty: return []
    result=[]
    for target,part in d.groupby("target_user"):
        s=part.set_index("timestamp").sort_index().resample(window).size()
        if len(s):
            mx=int(s.max())
            if mx>=3:
                result.append({"target_user":target,"max_abusive_in_window":mx,"window":window})
    return sorted(result,key=lambda x:x["max_abusive_in_window"],reverse=True)

def _metrics(y_true, prob, pred):
    out={"accuracy":float(accuracy_score(y_true,pred)),
         "precision":float(precision_score(y_true,pred,zero_division=0)),
         "recall":float(recall_score(y_true,pred,zero_division=0)),
         "f1":float(f1_score(y_true,pred,zero_division=0)),
         "confusion_matrix":confusion_matrix(y_true,pred,labels=[0,1]).tolist()}
    if len(set(y_true))==2:
        out["roc_auc"]=float(roc_auc_score(y_true,prob))
        out["pr_auc"]=float(average_precision_score(y_true,prob))
    else:
        out["roc_auc"]=None; out["pr_auc"]=None
    return out

class GCN(nn.Module):
    def __init__(self, in_dim, hidden=32):
        super().__init__(); self.w1=nn.Linear(in_dim,hidden); self.w2=nn.Linear(hidden,2)
    def forward(self,x,adj):
        h=torch.sparse.mm(adj,x); h=F.relu(self.w1(h))
        h=torch.sparse.mm(adj,h); return self.w2(h)

class GAT(nn.Module):
    def __init__(self,in_dim,hidden=32):
        super().__init__(); self.lin=nn.Linear(in_dim,hidden,bias=False)
        self.att=nn.Parameter(torch.empty(2*hidden)); nn.init.xavier_uniform_(self.att.view(1,-1))
        self.out=nn.Linear(hidden,2)
    def forward(self,x,edge_index,n):
        z=self.lin(x); src,dst=edge_index
        pair=torch.cat([z[src],z[dst]],dim=1)
        e=F.leaky_relu((pair*self.att).sum(dim=1),0.2)
                                                                          
                                                                        
        max_per_dst=torch.full((n,),-torch.inf,device=e.device,dtype=e.dtype)
        max_per_dst.scatter_reduce_(0,dst,e,reduce="amax",include_self=True)
        exp_e=torch.exp(e-max_per_dst[dst])
        denom=torch.zeros((n,),device=e.device,dtype=e.dtype)
        denom.index_add_(0,dst,exp_e)
        alpha=exp_e/denom[dst].clamp(min=1e-12)
        agg=torch.zeros_like(z)
        agg.index_add_(0,dst,z[src]*alpha.unsqueeze(1))
        return self.out(F.elu(agg+z))

class GraphSAGE(nn.Module):
    def __init__(self,in_dim,hidden=32):
        super().__init__(); self.self1=nn.Linear(in_dim,hidden); self.neigh1=nn.Linear(in_dim,hidden)
        self.self2=nn.Linear(hidden,2); self.neigh2=nn.Linear(hidden,2)
    def aggregate(self,x,edge_index,n):
        src,dst=edge_index; agg=torch.zeros_like(x); count=torch.zeros((n,1),device=x.device)
        agg.index_add_(0,dst,x[src]); count.index_add_(0,dst,torch.ones((len(dst),1),device=x.device))
        return agg/count.clamp(min=1)
    def forward(self,x,edge_index,n):
        h=F.relu(self.self1(x)+self.neigh1(self.aggregate(x,edge_index,n)))
        return self.self2(h)+self.neigh2(self.aggregate(h,edge_index,n))

def _adjacency(nodes, G):
    idx={n:i for i,n in enumerate(nodes)}
    edges=list(G.edges())
                                                                                    
    pairs=set()
    for u,v in edges:
        pairs.add((idx[u],idx[v])); pairs.add((idx[v],idx[u]))
    for i in range(len(nodes)): pairs.add((i,i))
    if not pairs: pairs={(i,i) for i in range(len(nodes))}
    ii=torch.tensor([p[0] for p in pairs],dtype=torch.long); jj=torch.tensor([p[1] for p in pairs],dtype=torch.long)
    n=len(nodes); deg=torch.zeros(n); deg.index_add_(0,ii,torch.ones(len(ii)))
    norm=deg.clamp(min=1).rsqrt()
    vals=norm[ii]*norm[jj]
    adj=torch.sparse_coo_tensor(torch.stack([ii,jj]),vals,(n,n)).coalesce()
    return adj,torch.stack([ii,jj]),idx

def train_models(G, table, X, epochs=60):
    y_raw=table.label.to_numpy()
    labelled=np.where(pd.notna(y_raw))[0]
    if len(labelled)<8 or len(set(y_raw[labelled].astype(int)))<2:
        raise ValueError("At least 8 labelled users and both classes are required for supervised model comparison. Upload ground-truth user labels or use the synthetic demo.")
    y=np.nan_to_num(y_raw,nan=0).astype(int)
    train_idx,test_idx=train_test_split(labelled,test_size=0.25,random_state=17,stratify=y[labelled])
    x=torch.tensor(X,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.long)
    adj,edge_index,index=_adjacency(list(table.index),G)
    results={}
    for name,model in [
        ("GCN",GCN(x.shape[1])),("GAT",GAT(x.shape[1])),("GraphSAGE",GraphSAGE(x.shape[1]))
    ]:
        torch.manual_seed(17)
        model=model
        opt=torch.optim.Adam(model.parameters(),lr=0.01,weight_decay=5e-4)
        loss_history=[]
        model.train()
        for epoch in range(max(5,min(int(epochs),200))):
            opt.zero_grad()
            if name=="GCN": logits=model(x,adj)
            else: logits=model(x,edge_index,len(table))
            loss=F.cross_entropy(logits[train_idx],yt[train_idx])
            loss.backward(); opt.step()
            loss_history.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            logits=model(x,adj) if name=="GCN" else model(x,edge_index,len(table))
            prob=torch.softmax(logits,dim=1)[:,1].numpy()
        pred=(prob>=0.5).astype(int)
        results[name]={**_metrics(y[test_idx],prob[test_idx],pred[test_idx]),
                       "epochs":len(loss_history),"final_train_loss":loss_history[-1],
                       "loss_history":loss_history}
    rf=RandomForestClassifier(n_estimators=160,class_weight="balanced",random_state=17)
    rf.fit(X[train_idx],y[train_idx])
    prob=rf.predict_proba(X[test_idx])[:,1]; pred=(prob>=0.5).astype(int)
    results["Random Forest"]={**_metrics(y[test_idx],prob,pred),"epochs":None,"final_train_loss":None,"loss_history":[]}
    results["_split"]={"train_users":len(train_idx),"test_users":len(test_idx),"labelled_users":len(labelled),
                       "seed":17,"note":"Random node split; graph-aware and temporal leakage checks are required for final research evaluation."}
    return results

def explain_user(user, G, table, community_map, repeated):
    if user not in G: return {"user_id":user,"message":"User not found in current graph."}
    row=table.loc[user]
    incoming=sorted(G.in_edges(user,data=True),key=lambda x:x[2]["weight"],reverse=True)[:5]
    outgoing=sorted(G.out_edges(user,data=True),key=lambda x:x[2]["weight"],reverse=True)[:5]
    target=next((x for x in repeated if x["target_user"]==user),None)
    evidence=[]
    if row.abuse_received_rate>=0.5: evidence.append("At least half of labelled incoming interactions are marked abusive.")
    if row.abuse_sent_rate>=0.5: evidence.append("At least half of labelled outgoing interactions are marked abusive.")
    if target: evidence.append(f"{target['distinct_sources']} distinct accounts sent {target['abusive_interactions']} labelled abusive interactions to this user.")
    if row.pagerank>=table.pagerank.quantile(.9): evidence.append("PageRank is in the top decile of this graph.")
    if row.betweenness>=table.betweenness.quantile(.9): evidence.append("Betweenness is in the top decile of this graph.")
    if not evidence: evidence.append("No configured high-risk rule was triggered; this is not proof that no harassment occurred.")
    return {"user_id":user,"community":community_map.get(user),
            "features":{k:float(row[k]) for k in table.columns if k!="label"},
            "incoming": [{"user":u,"weight":d["weight"],"abusive_count":d["abusive_count"]} for u,_,d in incoming],
            "outgoing": [{"user":v,"weight":d["weight"],"abusive_count":d["abusive_count"]} for _,v,d in outgoing],
            "evidence":evidence,"disclaimer":"Analytical indicators only. Not a finding of guilt; human review is required."}
