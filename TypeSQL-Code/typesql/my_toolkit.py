import re
import io
import json
import numpy as np


#  ----------------------------  Training Dataset From Data_Folder  ----------------------------

def load_data(sql_paths, table_paths ):
    if not isinstance(sql_paths, list):
        sql_paths = (sql_paths, )
    if not isinstance(table_paths, list):
        table_paths = (table_paths, )
    sql_data = []
    table_data = {}

    max_col_num = 0
    for SQL_PATH in sql_paths:
        print("    From %s"%SQL_PATH)
        with open(SQL_PATH) as inf:
            for idx, line in enumerate(inf):
                sql = json.loads(line.strip())
                sql_data.append(sql)

    for TABLE_PATH in table_paths:
        print("    From", TABLE_PATH)
        with open(TABLE_PATH) as inf:
            for line in inf:
                tab = json.loads(line.strip())
                table_data[tab[u'id']] = tab

    for sql in sql_data:
        assert sql[u'table_id'] in table_data
    return sql_data, table_data



#  ----------------------------  Load Embeddings From Glove_&_Paragram_sl999_czeng  ----------------------------

#Combinde Glove and Paragram_sl999_czeng embeddings, both of 1x300D
def load_concat_wemb(fn1, fn2 ):
    wemb1 = load_word_emb(fn1)  #Glove
    wemb2 = load_para_wemb(fn2) #Czeng
    backup = np.zeros(300, dtype=np.float32)
    print ('Embeddings')
    print ('    From', fn1," &")
    print ('    From', fn2)
    #New_Embedding(600D)= Glove(300D) + Paragram_sl999_czeng(300D) 
    comb_emb = {k: np.concatenate((wemb1.get(k, backup), wemb2.get(k, backup)), axis=0) for k in set(wemb1) | set(wemb2)}
    return None, None, comb_emb


#Function for Loading Paragram_sl999_czeng 
def load_para_wemb(file_name):
    f = open(file_name, 'r', encoding='utf-8')
    lines = f.readlines()
    ret = {}
    if len(lines[0].split()) == 2:
        lines.pop(0)
    for (n,line) in enumerate(lines):
        info = line.strip().split(' ')
        if info[0].lower() not in ret:
            ret[info[0].lower()] = np.array(info[1:]).astype(float)
    return ret


#Function for Loading Glove.6B.300d   
def load_word_emb(file_name, load_used=False):
    if not load_used:
        ret = {}
        with open(file_name,encoding="utf-8",mode="r") as inf:
            for idx, line in enumerate(inf):
                info = line.strip().split(' ')
                if info[0].lower() not in ret:
                    ret[info[0]] = np.array(info[1:]).astype(float)
        return ret
    else:
        print ('Load used word embedding')
        with open('glove/word2idx.json',mode="r",encoding="utf-8") as inf:
            w2i = json.load(inf)
        with open('glove/usedwordemb.npy',mode="r",encoding="utf-8") as inf:
            word_emb_val = np.load(inf)
        return w2i, word_emb_val
        
        
#Input load_word_and_type_emb [fn1, fn2]=[Glove,Paragram_sl999_czeng], data and content availability
def load_word_and_type_emb(fn1, fn2, sql_data, table_data, db_content, is_list=False, use_htype=False ):
    word_to_idx = {'<UNK>':0, '<BEG>':1, '<END>':2}
    word_num = 3
    N_word = 300
    embs = [np.zeros(N_word, dtype=np.float32) for _ in range(word_num)]
    _, _, word_emb = load_concat_wemb(fn1, fn2 )

    if is_list:
        for sql in sql_data:
            if db_content == 0:
                qtype = [[x] for x in sql["question_type_org_kgcol"]]
            else:
                qtype = sql['question_type_concol_list']
            for tok_typl in qtype:
                tys = " ".join(sorted(tok_typl))
                if tys not in word_to_idx:
                    emb_list = []
                    ws_len = len(tok_typl)
                    for w in tok_typl:
                        if w in word_emb:
                            emb_list.append(word_emb[w][:N_word])
                        else:
                            emb_list.append(np.zeros(N_word, dtype=np.float32))
                    word_to_idx[tys] = word_num
                    word_num += 1
                    embs.append(sum(emb_list) / float(ws_len))

        if use_htype:
            for tab in table_data.values():
                for col in tab['header_type_kg']:
                    cts = " ".join(sorted(col))
                    if cts not in word_to_idx:
                        emb_list = []
                        ws_len = len(col)
                        for w in col:
                            if w in word_emb:
                                emb_list.append(word_emb[w][:N_word])
                            else:
                                emb_list.append(np.zeros(N_word, dtype=np.float32))
                        word_to_idx[cts] = word_num
                        word_num += 1
                        embs.append(sum(emb_list) / float(ws_len))

    else:
        for sql in sql_data:
            if db_content == 0:
                qtype = sql['question_tok_type']
            else:
                qtype = sql['question_type_concol_list']
            for tok in qtype:
                if tok not in word_to_idx:
                    word_to_idx[tok] = word_num
                    word_num += 1
                    embs.append(word_emb[tok][:N_word])

        if use_htype:
            for tab in table_data.values():
                for tok in tab['header_type_kg']:
                    if tok not in word_to_idx:
                        word_to_idx[tok] = word_num
                        word_num += 1
                        embs.append(word_emb[tok][:N_word])


    agg_ops = ['null', 'maximum', 'minimum', 'count', 'total', 'average']
    for tok in agg_ops:
        if tok not in word_to_idx:
            word_to_idx[tok] = word_num
            word_num += 1
            embs.append(word_emb[tok][:N_word])

    emb_array = np.stack(embs, axis=0)
    return (word_to_idx, emb_array, word_emb)

       

#  ----------------------------  Training   ----------------------------


# "table_id": "2-12207755-6" // The specific table
# "sql": {"agg": 0, "sel": 0, "conds": [[5, 0, "30-31"]]}  //query-solution
def to_batch_query(sql_data, idxes, st, ed):
    query_gt = []; table_ids = []
    for i in range(st, ed):
        query_gt.append(sql_data[idxes[i]]['sql'])
        table_ids.append(sql_data[idxes[i]]['table_id'])
    return query_gt, table_ids
    
    
    
def to_batch_seq(sql_data, table_data, idxes, st, ed, db_content=0, ret_vis_data=False):
    q_seq = []; col_seq=[]; col_num=[] ;ans_seq=[]
    query_seq = []; gt_cond_seq = []; vis_seq = []

    q_type = []; col_type = []
    
    for i in range(st, ed):
        sql = sql_data[idxes[i]]
        if db_content == 0:
            q_seq.append([[x] for x in sql['question_tok']])
            q_type.append([[x] for x in sql["question_type_org_kgcol"]])
        else:
            q_seq.append(sql['question_tok_concol'])
            q_type.append(sql["question_type_concol_list"])
            
        col_type.append(table_data[sql['table_id']]['header_type_kg'])
        col_seq.append(table_data[sql['table_id']]['header_tok'])
        col_num.append(len(table_data[sql['table_id']]['header']))
        
        ans_seq.append((sql['sql']['agg'], sql['sql']['sel'], len(sql['sql']['conds']), #number of conditions + selection
            tuple(x[0] for x in sql['sql']['conds']), #col num rep in condition
            tuple(x[1] for x in sql['sql']['conds']))) #op num rep in condition, then where is str in cond?
            
        query_seq.append(sql['query_tok']) # real query string toks
        gt_cond_seq.append(sql['sql']['conds']) # list of conds (a list of col, op, str)
        vis_seq.append((sql['question'],
            table_data[sql['table_id']]['header'], sql['query'], [[x] for x in sql['question_tok']]))
    if ret_vis_data:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type, vis_seq
    else:
        return q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type
        
        
def Best_model_name(par ):
    new_data = 'old'; mode = 'sqlnet'
    use_emb =  ''
        
    print(par[1] , type(par[1]))

    agg_model_name = par[1] + '/%s_%s%s.agg_model'%(new_data, mode, use_emb)
    sel_model_name = par[1] + '/%s_%s%s.sel_model'%(new_data, mode, use_emb)
    cond_model_name =par[1] + '/%s_%s%s.cond_model'%(new_data, mode, use_emb)

    agg_embed_name = par[1] + '/%s_%s%s.agg_embed'%(new_data, mode, use_emb)
    sel_embed_name = par[1] + '/%s_%s%s.sel_embed'%(new_data, mode, use_emb)
    cond_embed_name =par[1] + '/%s_%s%s.cond_embed'%(new_data, mode, use_emb)

    print(agg_model_name, sel_model_name, cond_model_name)
    print(agg_embed_name, sel_embed_name, cond_embed_name)
    
    return agg_model_name, sel_model_name, cond_model_name, agg_embed_name, sel_embed_name, cond_embed_name

def best_model_n(args, for_load=False):
    new_data = 'old'
    mode = 'sqlnet'
    if for_load:
        use_emb = ''
    else:
        use_emb = '_train_emb' if args.train_emb else ''
        
    print(args.sd, type(args.sd))

    agg_model_name = args.sd + '/%s_%s%s.agg_model'%(new_data,
            mode, use_emb)
    sel_model_name = args.sd + '/%s_%s%s.sel_model'%(new_data,
            mode, use_emb)
    cond_model_name = args.sd + '/%s_%s%s.cond_model'%(new_data,
            mode, use_emb)

    agg_embed_name = args.sd + '/%s_%s%s.agg_embed'%(new_data, mode, use_emb)
    sel_embed_name = args.sd + '/%s_%s%s.sel_embed'%(new_data, mode, use_emb)
    cond_embed_name = args.sd + '/%s_%s%s.cond_embed'%(new_data, mode, use_emb)
    
    
    print(agg_model_name, sel_model_name, cond_model_name)
    print(agg_embed_name, sel_embed_name, cond_embed_name)
    
    return agg_model_name, sel_model_name, cond_model_name,\
           agg_embed_name, sel_embed_name, cond_embed_name