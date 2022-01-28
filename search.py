# from multiprocessing import Pool
import os
import sys
import time


def run_model(dataset, root, num_train, num_query, gpu, code_length,topk, num_class,threshold, eta, eta_1, eta_2, eta_3, batch_size):
    cmd = "python run.py"
    set = cmd + " --dataset " + dataset + " --root " + root + " --num-query " + str(num_query) + " --num-train " + \
          str(num_train) + " --gpu " + gpu +" --train " +" --code-length " +str(code_length) + " --topk " +str(topk) + \
              " --num_class "+str(num_class)+" --threshold " +str(threshold) + " --eta " +str(eta)+ " --batch-size " +str(batch_size) + \
                  " --eta_1 " +str(eta_1) + " --eta_2 " +str(eta_2)+ " --eta_3 " +str(eta_3)
    print(set)
    os.system(set)
    
#" --code-length 32 --prun " + "16,8,4 " + \


if __name__ == '__main__':
    ind = int(sys.argv[1]) #which dataset to train
    gpu = sys.argv[2] # which gpu to use
    dataset_list = ['flickr25k', 'nus-wide-tc-10', 'cifar-10']
    dataset = dataset_list[ind]
    root_dict = {'flickr25k': '../../datasets/flickr25k', 'nus-wide-tc-10': '../../datasets/NUS-WIDE', 'cifar-10': '../../datasets'}
    root = root_dict[dataset]
    # num_query_dict = {'flickr25k': 2000, 'nus-wide-tc-10': 5000, 'cifar-10':10000}
    # num_query = num_query_dict[dataset]
    # num_train_dict = {'flickr25k': 5000, 'nus-wide-tc-10': 10500, 'cifar-10':5000}
    # num_train = num_train_dict[dataset]
    # topk_dict = {'flickr25k': -1, 'nus-wide-tc-10': -1, 'cifar-10':-1}
    num_query_dict = {'flickr25k': 2000, 'nus-wide-tc-10': 5000, 'cifar-10':10000}
    num_query = num_query_dict[dataset]
    num_train_dict = {'flickr25k': 10000, 'nus-wide-tc-10': 5000, 'cifar-10':5000}
    num_train = num_train_dict[dataset]
    topk_dict = {'flickr25k': 5000, 'nus-wide-tc-10': 5000, 'cifar-10':-1}
    topk = topk_dict[dataset]
    code_length_list = [128]
    # eta_1_list = [0.1,0.5]
    # eta_2_list = [0]
    eta_list = [1] #这个参数不动
    #eta_1_list = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    #eta_1_list = [0, 0.1, 0.2]
    #eta_1_list = [0.3,0.4,0.5]
    #eta_1_list = [0.6,0.7,0.8]
    eta_1_list = [1]
    eta_2_list = [0.5]
    eta_3_list = [0]
    batch_size = 48
    
    # threshold_list=[0.1]
    num_class_list=[70]
    # threshold_list=[0.05,0.06,0.07]
    # threshold_list = [0.08,0.09,0.1]
    # threshold_list = [0.11,0.12,0.13]
    # threshold_list = [0.14,0.15, 0.16]
    # threshold_list = [0.17,0.18,0.19]
    # threshold_list = [0.20]
    threshold_list = [0.1]
    # num_class_list = [70]
    # pool = Pool(processes=1)


  
    print("start run all models")
    
    for code_length in code_length_list:
        for num_class in num_class_list:
            for threshold in threshold_list: 
                # for eta_1 in eta_1_list:
                #     for eta_2 in eta_2_list:
                for eta in eta_list:
                    for eta_1 in eta_1_list:
                        for eta_2 in eta_2_list:
                            for eta_3 in eta_3_list:
                                time.sleep(2)
                                run_model(dataset, root, num_train, num_query, gpu,code_length,topk, num_class, threshold,eta, eta_1, eta_2, eta_3, batch_size)
    # pool.close()
    # pool.join()