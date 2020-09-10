import json
import torch
import datetime
import argparse
import numpy as np
from typesql.training import *
from typesql.my_toolkit import *
from typesql.model.sqlnet import SQLNet
from typesql.lib.dbengine import DBEngine


import sys

def Print_Acc(ACC, SetCat="Val_"):
    print(SetCat+"Accuracy", ACC[0])
    print("    $agg:", np.round(ACC[1][0],3))
    print("    $sel:", np.round(ACC[1][1],3))
    print("    cond:", np.round(ACC[1][2],3)  , "[$cond_num, $cond_col,$cond_op,$cond_val]", np.round(ACC[1][3:], 3) )
    print()


    
if __name__ == '__main__':

    #  ************   Initialization   ************
    Flag=False; parameters=sys.argv
    USE_SMALL=False; GPU=True; BATCH_SIZE=64
    N_word=600; B_word=42; glove="glove.6B.300d.txt"
    
    Cont0_UNITS=180    # Manually set the number of units For Cont0
    Cont1_UNITS=120    # Manually set the number of units For Cont1
    
    
    
    print()
    print(" ----------------------------  Description  ---------------------------- ");print()
    print("    Run: python Testing.py       # Default Model (Epochs:7, Units:180, Opt:Adam )");
    print("    Run: python Testing.py Cont0 # If you have Already trained a model with Cont0")
    print("    Run: python Testing.py Cont1 # If you have Already trained a model with Cont1")
    print()
    print("Note: If you Test your Model (not Default) change the variables in this script manually"); print()
    
    
    print("Load data...")
    train_sql_data, table_data = load_data('data/train_tok.jsonl','data/train_tok.tables.jsonl')
    val_sql_data, val_table_data = load_data('data/dev_tok.jsonl','data/dev_tok.tables.jsonl')
    test_sql_data, test_table_data = load_data('data/test_tok.jsonl','data/test_tok.tables.jsonl')
    TRAIN_DB = 'data/train.db'; DEV_DB = 'data/dev.db'; TEST_DB = 'data/test.db';

    
    if(len(parameters)==1):
        print("    Testing: DEFAULT MODEL")
    elif (parameters[1]=="Cont0".lower()):
        print("    Testing: Your MODEL")
    elif(parameters[1]=="Cont1".lower()):
        Flag=True
        print("    Testing: Your MODEL")
    print('\n')
    
    if (not Flag ):
        
        print('\n') 
        print(" ----------------------------  Test Cont-0  ---------------------------- ")
        
        TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)
    
        # Give manually the path to the saved Models.
        agg_m ="Sav_Models/Saved_Model_Con0/epoch7.agg_model"
        sel_m ="Sav_Models/Saved_Model_Con0/epoch7.sel_model" 
        cond_m="Sav_Models/Saved_Model_Con0/epoch7.cond_model"
        agg_e ="Sav_Models/Saved_Model_Con0/epoch7.agg_embed" 
        sel_e ="Sav_Models/Saved_Model_Con0/epoch7.sel_embed"
        cond_e="Sav_Models/Saved_Model_Con0/epoch7.cond_embed"
    
        word_emb = load_word_and_type_emb('glove/'+glove, "para-nmt-50m/data/paragram_sl999_czeng.txt",\
                                             val_sql_data, val_table_data, 0, is_list=True, use_htype=False)
        model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb = False, db_content=0, N_h=Cont0_UNITS)
    
    
        #agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_n(args)
        print("    Loading from", agg_m)
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print("    Loading from", sel_m)
        model.selcond_pred.load_state_dict(torch.load(sel_m))
        print("    Loading from", cond_m)
        model.op_str_pred.load_state_dict(torch.load(cond_m))
        #only for loading trainable embedding
        print("    Loading from", agg_e)
        model.agg_type_embed_layer.load_state_dict(torch.load(agg_e))
        print("    Loading from", sel_e)
        model.sel_type_embed_layer.load_state_dict(torch.load(sel_e))
        print("    Loading from", cond_e)
        model.cond_type_embed_layer.load_state_dict(torch.load(cond_e))


        print()
        dev_acc=epoch_acc( model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY, 0)
        Print_Acc(dev_acc, "dev_acc")
                
        dev_exec=epoch_exec_acc(model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB, 0)
        print("Dev execution acc: ", dev_exec)
                
                
                
        ep_ac=epoch_acc( model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY, 0)
        Print_Acc(ep_ac, "ep_ac")
                
        epoch_exec=epoch_exec_acc(   model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB, 0)
        print("Test execution acc:",epoch_exec)
        
    else:
        print('\n') 
        print(" ----------------------------  Test Cont-1  ---------------------------- ")
        
        TEST_ENTRY=(True, True, True)  # (AGG, SEL, COND)
        # Give manually the path to the saved Models.
        agg_m ="Sav_Models/Saved_Model_Con1/old_sqlnet.agg_model"
        sel_m ="Sav_Models/Saved_Model_Con1/epoch7.sel_model" 
        cond_m="Sav_Models/Saved_Model_Con1/epoch7.cond_model"
        agg_e ="Sav_Models/Saved_Model_Con1/old_sqlnet.agg_embed" 
        sel_e ="Sav_Models/Saved_Model_Con1/epoch7.sel_embed"
        cond_e="Sav_Models/Saved_Model_Con1/epoch7.cond_embed"
        
        word_emb = load_concat_wemb('glove/'+glove, "para-nmt-50m/data/paragram_sl999_czeng.txt")
        model = SQLNet(word_emb, N_word=N_word, gpu=GPU, trainable_emb = False, db_content=1, N_h=Cont1_UNITS)

        #agg_m, sel_m, cond_m, agg_e, sel_e, cond_e = best_model_n(args)
        print("    Loading from", agg_m)
        model.agg_pred.load_state_dict(torch.load(agg_m))
        print("    Loading from", sel_m)
        model.selcond_pred.load_state_dict(torch.load(sel_m))
        print("    Loading from", cond_m)
        model.op_str_pred.load_state_dict(torch.load(cond_m))
        #For loading trainable embedding
        print("    Loading from", agg_e)
        model.agg_type_embed_layer.load_state_dict(torch.load(agg_e))
        print("    Loading from", sel_e)
        model.sel_type_embed_layer.load_state_dict(torch.load(sel_e))
        print("    Loading from", cond_e)
        model.cond_type_embed_layer.load_state_dict(torch.load(cond_e))
        
    
        print()
        dev_acc=epoch_acc( model, BATCH_SIZE, val_sql_data, val_table_data, TEST_ENTRY, 0)
        Print_Acc(dev_acc, "dev_acc")
    
        dev_exec=epoch_exec_acc(model, BATCH_SIZE, val_sql_data, val_table_data, DEV_DB, 0)
        print("Dev execution acc: ", dev_exec)
                
      
        ep_ac=epoch_acc( model, BATCH_SIZE, test_sql_data, test_table_data, TEST_ENTRY, 0)
        Print_Acc(ep_ac, "ep_ac")
                
        epoch_exec=epoch_exec_acc(   model, BATCH_SIZE, test_sql_data, test_table_data, TEST_DB, 0)
        print("Test execution acc:",epoch_exec)
            

