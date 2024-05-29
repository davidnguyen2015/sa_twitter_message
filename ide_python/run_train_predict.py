# import train
import sys
import subprocess


if __name__ == "__main__":

    # arg = "0.03"
    # train.run_tran("train-small", arg)

    # l2_reg_lambda
    # array_agr = "500.0,1000.0"
    #select: 0.0 -> 1.0

    # log_num_epochs
    #array_agr = "50,100,200,300,500,1000,2000"
    # select: 200 -> 2000

    # log_hidden_unit_num_filters
    #array_agr = "50,100,200,300,500,1000"
    # select: 128 -> 200

    # run for 4 model
    # CNN: run model CNN -> maxpool -> predict
    # RNN: run model CNN -> ouput -> predict
    # NN: Run CNN -> maxpool -> RNN -> outputt
    # Mix: Run CNN -> maxpool, RNN -> output, merge 2 result -> predict
    array_agr = "Merge_NN_Seq"

    list_arg = list(map(str, array_agr.split(",")))
    log_file = "result_20170805"

    # folder_data = "./data/data_1/"
    # for i in range(len(list_arg)):
    #     arg_value = list_arg[i]
    #     file = "./train_orther.py"
    #     if arg_value == "Merge_NN_Pra":
    #         file = "./train.py"
    #     subprocess.call([sys.executable, file, folder_data, log_file, str(arg_value)])

    # folder_data = "./data/data_2/"
    # for i in range(len(list_arg)):
    #     arg_value = list_arg[i]
    #     file = "./train_orther.py"
    #     if arg_value == "Merge_NN_Pra":
    #         file = "./train.py"
    #     subprocess.call([sys.executable, file, folder_data, log_file, str(arg_value)])

    # folder_data = "./data/data_3/"
    # for i in range(len(list_arg)):
    #     arg_value = list_arg[i]
    #     file = "./train_orther.py"
    #     if arg_value == "Merge_NN_Pra":
    #         file = "./train.py"
    #     subprocess.call([sys.executable, file, folder_data, log_file, str(arg_value)])

    folder_data = "./data/data_test/"
    for i in range(len(list_arg)):
        arg_value = list_arg[i]
        file = "./train_orther.py"
        if arg_value == "Merge_NN_Pra":
            file = "./train.py"
        subprocess.call([sys.executable, file, folder_data, log_file, str(arg_value)])