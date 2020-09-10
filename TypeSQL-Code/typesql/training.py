import json
import time
import torch
import datetime
import numpy as np
from .my_toolkit import *
from typesql.model.sqlnet import SQLNet
from typesql.lib.dbengine import DBEngine


# Displays and explains the arguments in command line
def description():
  print()
  print("--------  description  --------  ");print()
  
  print("!python mytypesql.py: Description"); print()
  
  print("!python mytypesql.py arg1 arg2 ... arg6")
  print('arg1: Folder=["Folder_For_Saving_Models"]')
  print('arg2: content=[0 | 1]')
  print('arg3: GPU=[True | False]')
  print('arg4: Optimizer=[adam|sgd|adadelta]')
  print('arg5: No units =[ 60 | 120 | 180 ]')
  print('arg6: Testing=[ yes | no ]')
  print()
  
  
  
def train(params, glove):
    print()
    
    #  ----------------------------  Initialization  ----------------------------
    N_word=600; B_word=42;  learning_rate = 1e-3; BATCH_SIZE=64; EPOCHS=7
    
    UNITS=int(params[5]); 
    if(params[3].lower()=="true"):
        GPU=True; 
    else:
        GPU=False;  
    
    print("GPU Available:", GPU)
    print("Numbe-Of-Units:",UNITS)
    if params[2].lower() == '0': 
        print("DB Content: Not_Available");
    else:
        print("DB Content: Available"); print()
    print()
    
    
    # $Slots filling problem (3 Models)
    # $Slots_Models: (Model_AGG, Model_SEL, Model_COND)
    TEST_ENTRY=(True, True, True)  
    TRAIN_ENTRY=(True, True, True)  
    
    
    TRAIN_AGG, TRAIN_SEL, TRAIN_COND = TRAIN_ENTRY # (AGG, SEL, COND)
    print("Training $Slots_Models:" ); print("    TRAIN_AGG=" ,TRAIN_AGG); 
    print("    TRAIN_SEL=",TRAIN_SEL); print("    TRAIN_COND", TRAIN_COND)
    print()
    
    
    #  ----------------------------  Load DataSets  ----------------------------
    #Data for training:(train_tok/train_tok.tables)
    #Data for testing:  (test_tok/test_tok.tables)
    #Data for evaluation;(dev_tok/dev_tok.tables)
    
    print("Load data...")
    train_sql_data, table_data = load_data('data/train_tok.jsonl','data/train_tok.tables.jsonl')
    val_sql_data, val_table_data = load_data('data/dev_tok.jsonl','data/dev_tok.tables.jsonl')
    test_sql_data, test_table_data = load_data('data/test_tok.jsonl','data/test_tok.tables.jsonl')
                
    TRAIN_DB = 'data/train.db'; DEV_DB = 'data/dev.db'; TEST_DB = 'data/test.db'; 
    print()
    
    #  ----------------------------  Models_Location  ----------------------------
    agg_m, sel_m, cond_m, agg_e, sel_e, cond_e= Best_model_name(params)
    
    print("Location:", params[1])
    print('    ', agg_m, sel_m, cond_m); print('    ', agg_e, sel_e, cond_e) ; print()
    
    
    #  ----------------------------  Load Glove_Embeddings & SQLNet_Model  ----------------------------
    #content=0: Training with not tables' entries 
    #content=1: Training with tables' entries available  

    train_emb=False
    if params[2].lower() == '0': 
        word_emb = load_word_and_type_emb('glove/'+glove, "para-nmt-50m/data/paragram_sl999_czeng.txt",\
                                                     val_sql_data, val_table_data, 0, is_list=True, use_htype=False )
        print()
        model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=train_emb, N_h=UNITS , db_content=0, )
    
    else: 
        word_emb = load_concat_wemb('glove/'+glove, "para-nmt-50m/data/paragram_sl999_czeng.txt" )
        print()
        model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb=train_emb, N_h=UNITS , db_content=1)
        
    print()
    if (params[4].lower()=="adam" ):
        print("Optimizer: ADAM")
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0)
    elif (params[4].lower()=="sgd" ):
        print("Optimizer: SGD")
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    elif (params[4].lower()=="adadelta"):
        print("Optimizer:  AdaDelta")
        optimizer = torch.optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
    print()


    
    #  ----------------------------  Training  ----------------------------
    print(" ---------------  Start Training  --------------- ")
    print("    train data",len(train_sql_data),"table len",len(table_data))
    print("    val.  data ",len(val_sql_data) , "table len",len(val_table_data))
    print("    test  data",len(test_sql_data), "table len",len(test_table_data))
    
    #initial accuracy
    init_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY, int(params[2]))
    print()
    
    
    best_agg_acc = init_acc[1][0]; best_agg_idx = 0
    best_sel_acc = init_acc[1][1]; best_sel_idx = 0
    best_cond_acc =init_acc[1][2]; best_cond_idx = 0

    print("Accuracy", init_acc[0])
    print("    $agg:", np.round(init_acc[1][0],3))
    print("    $sel:", np.round(init_acc[1][1],3))
    print("    cond:", np.round(init_acc[1][2],3)  , "[$cond_num, $cond_col,$cond_op,$cond_val]", np.round(init_acc[1][3:], 3) )


    
    if TRAIN_AGG:
        torch.save(model.agg_pred.state_dict(), agg_m)
        torch.save(model.agg_type_embed_layer.state_dict(), agg_e)
    if TRAIN_SEL:
        torch.save(model.selcond_pred.state_dict(), sel_m)
        torch.save(model.sel_type_embed_layer.state_dict(), sel_e)
    if TRAIN_COND:
        torch.save(model.op_str_pred.state_dict(), cond_m)
        torch.save(model.cond_type_embed_layer.state_dict(), cond_e)
    print()

    # Total validation
    Train_Loss=[]; Train_Acc=[]; Val_Acc=[]; Time=[]
    
    #Training Set 
    tr_agg=[]; tr_sel=[]; tr_cond=[]
    tr_num=[]; tr_col=[]; tr_opr=[]; tr_val=[]
    
    # Validation Set
    val_agg=[]; val_sel=[]; val_cond=[]
    val_num =[]; val_col =[]; val_opr =[]; val_val =[]; 



    for i in range(EPOCHS):
        print(" * * * * * * * * *   Epoch ", i+1,"  * * * * * * * * *")
        tic = time.clock()

        loss=epoch_train( model, optimizer, BATCH_SIZE, train_sql_data, table_data, TRAIN_ENTRY,  int(params[2]))
        Train_Loss.append(loss);  print(' Loss = ', loss)
        
        
        acc=epoch_acc( model, BATCH_SIZE, train_sql_data, table_data, TRAIN_ENTRY,  int(params[2]))
        Train_Acc.append(acc[0]);  tr_agg.append(acc[1][0])  ; tr_sel.append(acc[1][1]); tr_cond.append(acc[1][2])
        tr_num.append(acc[1][3]); tr_col.append(acc[1][4]); tr_opr.append(acc[1][5]); tr_val.append(acc[1][6])   
        print(); Print_Acc(acc, "Train");
       
        # Validation set 
        val_acc = epoch_acc(model, BATCH_SIZE, val_sql_data, val_table_data, TRAIN_ENTRY,  int(params[2]), False)
        Val_Acc.append(val_acc[0]);  val_agg.append(val_acc[1][0])  ; val_sel.append(val_acc[1][1]); val_cond.append(val_acc[1][2])
        val_num.append(val_acc[1][3]); val_col.append(val_acc[1][4]); val_opr.append(val_acc[1][5]); val_val.append(val_acc[1][6])  
        print(); Print_Acc(val_acc);
        
        # Training time for each epoch
        toc = time.clock()
        Time.append(toc - tic)
        print("Epoch duration:",toc, toc/60)

        if TRAIN_SEL:
            if val_acc[1][1] > best_sel_acc:
                best_sel_acc = val_acc[1][1]
                best_sel_idx = i+1
                torch.save(model.selcond_pred.state_dict(),
                    params[1] + '/epoch%d.sel_model%s'%(i+1, ''))
                torch.save(model.selcond_pred.state_dict(), sel_m)

                torch.save(model.sel_type_embed_layer.state_dict(),
                                params[1] + '/epoch%d.sel_embed%s'%(i+1, ''))
                torch.save(model.sel_type_embed_layer.state_dict(), sel_e)

        if TRAIN_COND:
            if val_acc[1][2] > best_cond_acc:
                best_cond_acc = val_acc[1][2]
                best_cond_idx = i+1
                torch.save(model.op_str_pred.state_dict(),
                    params[1] + '/epoch%d.cond_model%s'%(i+1, ''))
                torch.save(model.op_str_pred.state_dict(), cond_m)

                torch.save(model.cond_type_embed_layer.state_dict(),
                                params[1] + '/epoch%d.cond_embed%s'%(i+1, ''))
                torch.save(model.cond_type_embed_layer.state_dict(), cond_e)


               
    print('\n* * * * * * * * * * * *    Results    * * * * * * * * * * * *')
    
    print(params)
    print()
    
    print('Exec time:', sum(Time));print('Avg time:', sum(Time)/len(Time));print("Time=", np.round(Time,3))
    print('Exec time:', sum(Time)/60);print('Avg time:', (sum(Time)/len(Time))/60);print("Time=", np.round(Time,3)/60)
    print(); 
    
    print("Train_Loss=", np.round(Train_Loss,3)); 
    print("Train_Acc=", np.round(Train_Acc,3));  
    print("Val_Acc=", np.round(Val_Acc,3));
    
    print()
    print("tr_agg=", np.round(tr_agg,3) ); print("tr_sel=", np.round(tr_sel,3) ); print("tr_cond=", np.round(tr_cond,3) )
    print("tr_num=", np.round(tr_num,3) ); print("tr_col=", np.round(tr_col,3) ); print("tr_opr =",  np.round(tr_opr ,3) ); 
    print("tr_val=", np.round(tr_val,3) ); 
    print()
    print("val_agg=", np.round(val_agg,3) ); print("val_sel=", np.round(val_sel,3) ); print("val_cond=", np.round(val_cond,3) )
    print("val_num=", np.round(val_num,3) ); print("val_col=", np.round(val_col,3) ); print("val_opr =",  np.round(val_opr ,3) ); 
    print("val_val=", np.round(val_val,3) ); 

    print()
    print('Exec time:', sum(Time)/60);print('Avg time:', (sum(Time)/len(Time))/60);print("Time=", [a/60 for a in np.round(Time,3)] )

        
    if (params[6].lower()=="yes"):
        print();print(" ---------------  Testing  --------------- ")

        print("Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%epoch_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY,  int(params[2])))
            
        print("Test execution acc: %s"%epoch_exec_acc(
            model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB,  int(params[2])))
        
        
        ACC=epoch_acc( model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY,  int(params[2]))
        exec_Acc=epoch_exec_acc( model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB,  int(params[2]))
        
        print("Test acc_qm: %s;\n  breakdown on (agg, sel, where): %s"%ACC)
        print("Test execution acc: %s"%exec_Acc)













#   * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
#   * * * * * * * * * * * * * * * * * *         Training_Methods_For:         * * * * * * * * * * * * * * * * * *

def Print_Acc(ACC, SetCat="Val_"):
    print(SetCat+"Accuracy", ACC[0])
    print("    $agg:", np.round(ACC[1][0],3))
    print("    $sel:", np.round(ACC[1][1],3))
    print("    cond:", np.round(ACC[1][2],3)  , "[$cond_num, $cond_col,$cond_op,$cond_val]", np.round(ACC[1][3:], 3) )
    print()



# estimate the accuracy of each Epoch
def epoch_acc(model, batch_size, sql_data, table_data, pred_entry, db_content, error_print=False):
    # SQL keywords (agg,sel,cond)
    one_acc_num = 0.0 # Accuracy of each distict SQL keyword.
    tot_acc_num = 0.0 # Total acuuracy for  all SQL keywords. 
    perm= list(range(len(sql_data)))# Access data with indes.                       
    st = 0 # (st, ed) =(start, end) are defining  each batch.

    model.eval()
    
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        #Create the batch data
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type,\
         raw_data = to_batch_seq(sql_data, table_data, perm, st, ed, db_content, ret_vis_data=True)
         
        # The inputs to the model are:
        #     a) Tables header:  raw_col_seq= {... , "['Date', 'Opponent', 'Score', 'Loss', 'Attendance', 'Record']"}
        #     b) NL-question:  raw_q_seq= {... , "What's the original air date of the episode with a broadcast order s04 e01?", ...}
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        
        # Take the solution of each query and the table's id to which belong 
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        
        # Evaluate the model
        score = model.forward(q_seq, col_seq, col_num, q_type, col_type, pred_entry)
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, pred_entry)
        one_err, tot_err = model.check_acc(raw_data, pred_queries, query_gt, pred_entry, error_print)

        one_acc_num += (ed-st-one_err)
        tot_acc_num += (ed-st-tot_err)

        st = ed
    return tot_acc_num / len(sql_data), one_acc_num / len(sql_data)
    
    
    
    
def epoch_train(model, optimizer, batch_size, sql_data, table_data, pred_entry, db_content):
    model.train()
    perm=np.random.permutation(len(sql_data))
    cum_loss = 0.0; st = 0
    
    # (st, ed) =(start, end) are defining  each batch.
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)

        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type = \
                to_batch_seq(sql_data, table_data, perm, st, ed, db_content)
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        gt_sel_seq = [x[1] for x in ans_seq]
        gt_agg_seq = [x[0] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, q_type, col_type, pred_entry,
                gt_where=gt_where_seq, gt_cond=gt_cond_seq, gt_sel=gt_sel_seq)
        loss = model.loss(score, ans_seq, pred_entry, gt_where_seq)
        # cum_loss += loss.data.cpu().numpy()[0]*(ed - st)
        cum_loss += loss.data.cpu().numpy() * (ed - st)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        st = ed

    return cum_loss / len(sql_data)
    
def epoch_exec_acc(model, batch_size, sql_data, table_data, db_path, db_content):
    engine = DBEngine(db_path)
    model.eval()
    perm = list(range(len(sql_data)))
    tot_acc_num = 0.0
    acc_of_log = 0.0
    st = 0
    while st < len(sql_data):
        ed = st+batch_size if st+batch_size < len(perm) else len(perm)
        q_seq, col_seq, col_num, ans_seq, query_seq, gt_cond_seq, q_type, col_type, raw_data = \
            to_batch_seq(sql_data, table_data, perm, st, ed, db_content, ret_vis_data=True)
        raw_q_seq = [x[0] for x in raw_data]
        raw_col_seq = [x[1] for x in raw_data]
        gt_where_seq = model.generate_gt_where_seq(q_seq, col_seq, query_seq)
        query_gt, table_ids = to_batch_query(sql_data, perm, st, ed)
        gt_sel_seq = [x[1] for x in ans_seq]
        gt_agg_seq = [x[0] for x in ans_seq]
        score = model.forward(q_seq, col_seq, col_num, q_type, col_type, (True, True, True))
        pred_queries = model.gen_query(score, q_seq, col_seq,
                raw_q_seq, raw_col_seq, (True, True, True))

        for idx, (sql_gt, sql_pred, tid) in enumerate(
                zip(query_gt, pred_queries, table_ids)):
            ret_gt = engine.execute(tid, sql_gt['sel'], sql_gt['agg'], sql_gt['conds'])
            try:
                ret_pred = engine.execute(tid, sql_pred['sel'], sql_pred['agg'], sql_pred['conds'])
            except:
                ret_pred = None
            tot_acc_num += (ret_gt == ret_pred)

        st = ed

    return tot_acc_num / len(sql_data)
