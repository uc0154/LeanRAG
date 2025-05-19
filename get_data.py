import json
import os

from tqdm import tqdm
from prompt import PROMPTS
import tiktoken
from tools.utils import read_jsonl, write_jsonl, create_if_not_exist
from build_graph import deepseepk_model_if_cache as use_llm
from concurrent.futures import ThreadPoolExecutor, as_completed
import tiktoken
threshold=50
def summarize_entity(entity_name, description, summary_prompt, threshold, tokenizer):
    tokens = len(tokenizer.encode(description))
    if tokens > threshold:
        exact_prompt = summary_prompt.format(entity_name=entity_name, description=description)
        response = use_llm(exact_prompt)
        return entity_name, response
    return entity_name, description  # 不需要摘要则返回原始 description


def truncate_data():
    # relation_path="/cpfs04/user/zhangyaoze/workspace/trag/processed_data/relation.jsonl"
    # relation_output_path="/cpfs04/user/zhangyaoze/workspace/trag/late_data/relation.jsonl"
    # entity_path="/cpfs04/user/zhangyaoze/workspace/trag/processed_data/entity.jsonl"
    # entity_output_path="/cpfs04/user/zhangyaoze/workspace/trag/late_data/entity.jsonl"
    relation_path="processed_data/relation.jsonl"
    relation_output_path="data/relation.jsonl"
    entity_path="processed_data/entity.jsonl"
    entity_output_path="data/entity.jsonl"
    res=[]
    i=0
    with open(relation_path,"r") as f:
        for uline in f:
                line=json.loads(uline)
                res.append(line)
                i+=1
                if i==20000:
                    break
    write_jsonl(res,relation_output_path)
    
    res=[]
    i=0
    with open(entity_path,"r") as f:
        for uline in f:
                line=json.loads(uline)
                if "wtr20" in line['source_id']:
                    res.append(line)
                    i+=1
                    if i==20000:
                        break
    write_jsonl(res,entity_output_path)
def deal_duplicate_entity():
    relation_path="/cpfs04/user/zhangyaoze/workspace/Common_KG_Build/relation.jsonl"
    relation_output_path="/cpfs04/user/zhangyaoze/workspace/trag/processed_data/relation.jsonl"
    entity_path="/cpfs04/user/zhangyaoze/workspace/Common_KG_Build/entity.jsonl"
    entity_output_path="/cpfs04/user/zhangyaoze/workspace/trag/processed_data/entity.jsonl"
    
    all_entities=[]
    all_relations=[]
    e_dic={}
    r_dic={}
    summary_prompt=PROMPTS['summary_entities']
    with open(entity_path,"r")as f:
        for xline in f:
            line=json.loads(xline)
            entity_name=str(line['entity_name'])
            entity_type=line['entity_type']
            description=line['description']
            source_id=line['source_id']
            if entity_name not in e_dic.keys():
                e_dic[entity_name]=dict(
                    entity_name=str(entity_name),
                    entity_type=entity_type,
                    description=description,
                    source_id=source_id,
                    degree=0,
                )
            else:
                e_dic[entity_name]['description']+=" | "+ description
                if e_dic[entity_name]['source_id']!= source_id:
                    e_dic[entity_name]['source_id']+= "|"+source_id
                    
                  
                    
    tokenizer = tiktoken.get_encoding("cl100k_base")
    to_summarize = []
    for k,v in e_dic.items():
        v['source_id']="|".join(set(v['source_id'].split("|")))
        description=v['description']
        tokens = len(tokenizer.encode(description))
        if tokens > threshold:
            to_summarize.append((k, description))
        else:
            all_entities.append(v)
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(summarize_entity, k, desc, summary_prompt, threshold, tokenizer): k
            for k, desc in to_summarize
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="Summarizing descriptions"):
            k, summarized_desc = future.result()
            e_dic[k]['description'] = summarized_desc
            all_entities.append(e_dic[k])


    write_jsonl(all_entities,entity_output_path)
    with open(relation_path,"r")as f:
        for xline in f:
            line=json.loads(xline)
            src_tgt=str(line['src_tgt'])
            tgt_src=str(line['tgt_src'])
            description=line['description']
            weight=1
            source_id=line['source_id']
            r_dic[(src_tgt,tgt_src)]={
                'src_tgt':str(src_tgt),
                'tgt_src':str(tgt_src),
                'description':description,
                'weight':weight,
                'source_id':source_id
            }
            # e_dic[src_tgt]['degree']+=1
            # e_dic[tgt_src]['degree']+=1
    write_jsonl(all_relations,relation_output_path)
if __name__=="__main__":
    # deal_duplicate_entity()
    truncate_data()
