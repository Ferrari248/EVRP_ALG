import geatpy as ea
import numpy as np
from configparser import ConfigParser
import time
from utils import *


if __name__ == '__main__':
    """================================用户初始设置============================"""
    """
    方案1：DC。    11遗传， 15全排列
    方案2：NNI。   21遗传， 25全排列
    方案3：CAI。   31遗传， 32遗传缓存， 35全排列， 36全排列缓存 （不考虑车辆可重复利用） 
    """
    cp = ConfigParser()
    # 选择数据路径，并读取遗传算法配置文件；选择配置项
    dataPath = './datas/data01'
    cp.read(dataPath + '/ga.conf', encoding='UTF-8')
    alg = 32

    # 选择算法方案，并将数据路径传入问题类
    if alg == 11 or alg == 15:
        from MyProblem01 import *
        problem = MyProblem01(dataPath)
    elif alg == 21 or alg == 25:
        from MyProblem02 import *
        problem = MyProblem02(dataPath)
    elif 31 <= alg <= 32 or 35 <= alg <= 36:
        from MyProblem03 import *
        if alg == 31 or alg == 35:
            problem = MyProblem03(dataPath, use_cache=False)
        else:
            problem = MyProblem03(dataPath, use_cache=True)
    else:
        raise Exception("没有该算法")

    self = problem
    # raise Exception("pause")

    # 使用遗传算法的情况
    if alg % 10 < 5:
        """==================================种群设置=============================="""
        # 编码方式，采用排列编码
        Encoding = 'P'
        # 种群规模
        NIND = int(cp.get('ga', 'NIND'))
        # 创建区域描述器
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)
        # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        population = ea.Population(Encoding, Field, NIND)

        """================================算法参数设置============================="""
        # 实例化一个算法模板对象
        # SGA：普通    EGA：精英保留    SEGA：增强精英保留（需要交叉变异概率较大）
        myAlgorithm = ea.soea_SEGA_templet(problem, population)
        # 最大进化代数
        myAlgorithm.MAXGEN = int(cp.get('ga', 'MAXGEN'))
        # 变异概率
        myAlgorithm.mutOper.Pm = float(cp.get('ga', 'Pm'))
        # 设置每隔多少代记录日志，若设置成0则表示不记录日志
        myAlgorithm.logTras = 10
        # 设置是否打印输出日志信息
        myAlgorithm.verbose = True
        # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        myAlgorithm.drawing = 1

        """===========================调用算法模板进行种群进化========================"""
        # 执行算法模板，得到最优个体以及最后一代种群
        [BestIndi, population] = myAlgorithm.run()
        # 把最优个体的信息保存到文件中
        # BestIndi.save()

        """==================================输出结果=============================="""
        print('评价次数：%s' % myAlgorithm.evalsNum)
        print('算法执行耗时 %s 秒' % myAlgorithm.passTime)
        if BestIndi.sizes != 0:
            print('最佳目标函数值为：%s' % BestIndi.ObjV[0][0])
            best_journey = BestIndi.Phen[0, :]
            print("表现型长度： %d" % len(best_journey))
            print('最后表现型为：')
            for i in range(len(best_journey)):
                print(int(best_journey[i]), end=' ')
            print()
            # 最佳方案出图
            problem.show_final_ans(best_journey)
        else:
            print('没找到可行解。')

    # 使用全排列的情况
    else:
        t1 = time.time()
        minobjv, perm_res_list, feasible_count = problem.permute_solve()
        t2 = time.time()
        print("执行时间：%.3f 秒" % (t2 - t1))
        print("总共枚举出 %d 条可行解" % feasible_count)
        if minobjv == 999_999:
            print('没找到可行解。')
        else:
            print('最佳目标函数值为：%s' % minobjv)
            best_journey = perm_res_list
            print("表现型长度： %d" % len(best_journey))
            print('最后表现型为：')
            for i in range(len(best_journey)):
                print(int(best_journey[i]), end=' ')
            print()
            # 最佳方案出图
            problem.show_final_ans(best_journey)

