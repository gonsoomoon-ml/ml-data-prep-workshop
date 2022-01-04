from IPython.display import display as dp
def show_dic_items(dic_name, num=3):    
    '''
    사전 데이터의 앞의 일부 데이터를 보여 줌.
    '''
    for i, (k, v) in enumerate(dic_name.items()):
        print(k,v)
#         dp(k, v)
        if i == (num -1) :
            break
            



