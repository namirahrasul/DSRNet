import datetime
import os
import shutil
import time

#counts number of trainable parameters
#same as net_utils but different print
def count_parameters(model):
    number_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Parameters: {:d} or {:.2f}M'.format(number_params, number_params / (1024 * 1024))) #in millions
    return number_params


def contains(key, lst): #return true if lst contains the key
    flag = False
    for item in lst:
        if key == item:
            flag = True
    return flag


def make_empty_dir(new_dir): #deletes previous directory containing files and makes new empty one
    if os.path.exists(new_dir):
        shutil.rmtree(new_dir)
    os.makedirs(new_dir, exist_ok=True)


def get_timestamp():
    return str(time.time()).replace('.', '') #returns time as secondsmilliseconds vs seconds.milliseconds can be used timestamp identifier


def get_formatted_time(): #returns current time formatted as YYYYMMDD-HHMMSS
    return datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == '__main__': #if the script is being run directly (as opposed to being imported as a module).
    pass
