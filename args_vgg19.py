class Train_Args():
    is_training                 =       True
    boxsize                     =       180
    dim_x                       =       1919 # micrograph width
    dim_y                       =       1855
    name_length                 =       4
    name_prefix                 =       "ptsm_10_201124_grid4_"
    mic_path                    =       "./data/mic_training/"
    model_save_path             =       "./data/model/"  # end with /
    positive1_box_path          =       "./data/positive/"
    negative1_box_path          =       "./data/negative/"
    args_filename               =       "args_vgg19_19S.py"  # filename of this document
    model_filename              =       "vgg19.py"

    positive1_mic_start_num     =       1
    positive1_mic_end_num       =       99
    negative1_mic_start_num     =       1
    negative1_mic_end_num       =       125
    num_positive1               =       771  # number of positive particles
    num_negative1               =       1000

    do_train_again              =       False
    num_positive2               =       800
    num_negative2               =       800
    # positive2_box_path          =       "../data/19Sdata/sel_positive/"
    # negative2_box_path          =       "../data/19Sdata/sel_negative/"
    positive2_mic_start_num     =       30051
    positive2_mic_end_num       =       30090
    negative2_mic_start_num     =       30051
    negative2_mic_end_num       =       30100

    rotation_angel              =       90
    rotation_n                  =       4
    num_p_test                  =       855
    num_n_test                  =       734

    regularization              =       True
    reg_rate                    =       0.001
    dropout                     =       True
    dropout_rate                =       0.5


    learning_rate               =       0.00001   # !!!!

    batch_size                  =       100
    num_epochs                  =       100


    decay_rate                  =       0.95
    decay_step                  =       200 

class Predict_Args():
    is_training                 =       False
    data_path                   =       "/ldata/swzhang/vgg_ptsm/data/mic/"
    result_path                 =       "/ldata/swzhang/vgg_ptsm/data/result/"
    model_save_path             =       "/ldata/swzhang/vgg_ptsm/data/model_dc_lr0.00001_0.95_rr0.0005/"
    boxsize                     =       180
    start_mic_num               =       1
    end_mic_num                 =       3000
    dim_x                       =       1919
    dim_y                       =       1855
    edge_space                  =       6
    scan_step                   =       20    #!!!!
    accuracy                    =       0.8   ####  
    threhold                    =       0.5   ####
    name_length                 =       4
    name_prefix                 =       "ptsm_tmk63tub_220311_"
    name_suffix                 =       ""#"_patch_aligned"

    batch_size                  =       50
