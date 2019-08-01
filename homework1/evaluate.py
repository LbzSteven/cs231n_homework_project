def cal_time(f, *args):
    """
    函数功能: 计算函数f执行时间

    输入：
        f(function): 所要执行的函数
        *args: 可变长度的函数参数
    输出：
        time(int): 函数执行时间
    """
    import time

    t_st = time.time()
    f(*args)
    t_ed = time.time()

    return t_ed - t_st
