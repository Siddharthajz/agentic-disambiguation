import re
import json 
from typing import  Any, Dict
import numpy as np 
from collections import Counter, defaultdict
import math 
from functools import lru_cache 
from pyserini.search.lucene import LuceneSearcher 
from pyserini.index.lucene import IndexReader
from tqdm import tqdm 
from jnius import autoclass
import os


StandardAnalyzer = autoclass('org.apache.lucene.analysis.standard.StandardAnalyzer')
EnglishAnalyzer = autoclass('org.apache.lucene.analysis.en.EnglishAnalyzer')
CharArraySet = autoclass('org.apache.lucene.analysis.CharArraySet')
default_stopset = EnglishAnalyzer.getDefaultStopSet()
analyzer = StandardAnalyzer(default_stopset)
CharTermAttribute = autoclass('org.apache.lucene.analysis.tokenattributes.CharTermAttribute')
# os.environ["JAVA_TOOL_OPTIONS"] = (
#     "-Dpyserini.index.directory.impl=SimpleFSDirectory"
# )
# TOKEN_RE = re.compile(r"\w+", re.UNICODE)


#change this later
@lru_cache(maxsize=100_000)  
def get_term_prob(searcher, term, total_terms):
    """Return P(w|C) = cf / total_terms, cached."""
    try:
        reader = IndexReader(searcher.index_dir) 
        df, cf = reader.get_term_counts(term)
        if cf:
            return cf / total_terms
        else:
            return 1e-12
    except:
        return 1e-12


#look into more options for this
def tokenize(text: str):
    """Tokenize text with Lucene’s StandardAnalyzer"""
    ts = analyzer.tokenStream("field", text)  
    term_att = ts.addAttribute(CharTermAttribute)
    ts.reset()
    toks = []
    while ts.incrementToken():
        toks.append(term_att.toString())
    ts.end()
    ts.close()
    return toks


def load_json(path: str)->Dict[Any, Any]:
    """
    Loads a JSON file and returns its contents as a dictionary.

    Args:
        path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON contents.

    """
    
    try:
        with open(path, "r") as file:
            return json.load(file) or {}
    except FileNotFoundError as e:
        raise FileNotFoundError(f"JSON file not found: {path}") from e 
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Error parsing JSON file: {path}") from e
    except Exception as e:
         raise Exception(f"Unexpected error while loading JSON file: {path}\n{str(e)}") from e 


def save_json(data: Dict[Any, Any], path: str)-> None:
    """
    Writes data into a JSON file

    Args:
    dict: Parsed JSON contents.
    path (str): Path to the JSON file.
       

    Returns:
        None 

    """
    try: 
        with open(path, "w") as file:
            json.dump(data,file, indent=4)
    except TypeError as e:
        raise TypeError(f"Data contains non-serializable objects: {e}") from e
    except Exception as e:
        raise Exception(f"Unexpected error while saving JSON file: {path}\n{str(e)}") from e

def compute_softmax(scores):
    "Compute the softmax-normalized probabilities of a list or array of scores."
    scores = np.array(scores, dtype=np.float64)
    shifted = scores - np.max(scores)
    exp_scores = np.exp(shifted)
    return exp_scores / np.sum(exp_scores)


#refactor 
def clarity_score_from_results(searcher, results):
    """Compute clarity score for a Pyserini result JSON."""
   
    clarity_scores={}
    for result in tqdm(results):
        
        qid=result['question_id']

        #compute P(d|q)
        p_dq=compute_softmax([r['score'] for r in result['retrieved_docs']])
        


        #build doc models P(w|d)
        doc_models=[]
        for r in result['retrieved_docs']:
            text=r['text']
            tokens= tokenize(text)
            total = len(tokens)
            
            tf = Counter(tokens)
            p_wd = {w: c / total for w, c in tf.items()}
            doc_models.append(p_wd)

        
        p_wq = defaultdict(float)
        for p_wd, wgt in zip(doc_models, p_dq):
            for w, p in p_wd.items():
                p_wq[w] += wgt * p
        
        #calculate normalising constant 
        Z = sum(p_wq.values()) or 1.0
        #normalise 
        for w in list(p_wq.keys()):
            p_wq[w] /= Z
        
        index_path = searcher.index_dir

        reader = IndexReader(index_path)
        total_terms = reader.stats()['total_terms']
        p_wc = {w: get_term_prob(searcher, w, total_terms) for w in p_wq.keys()}


        #compute clarity score 
        clarity = 0.0
        for w, pwq in p_wq.items():
            pwc = p_wc[w]
            clarity += pwq * math.log2(pwq / pwc)
        clarity_scores[qid]= float(clarity)
       

    return clarity_scores




result_json_path='../results/sparse_rag_results.json'
sparse_index= "wikipedia-dpr"
searcher = LuceneSearcher.from_prebuilt_index(sparse_index, True)
results= load_json(result_json_path)['results']

#run this in batches
clarity_score_results=clarity_score_from_results(searcher,results[150:])
save_json(clarity_score_results, '../results/clarity_score_results_1.json')