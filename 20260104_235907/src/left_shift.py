import numpy as np

# 这个函数是作业车间调度（JSSP） 中核心的「工序左移合法性校验与起始时间计算」函数，
# 核心逻辑是：对指定工序 a，先计算其作业就绪时间和设备就绪时间，再判断是否能将该工序 “左移” 插入设备的空闲时段（而非直接放到设备任务队列末尾），
# 最终返回工序 a 的最优起始时间，以及是否成功左移的标记。
# a 待调度的工序编号 int
# durMat 工序时长矩阵（一维 / 二维），durMat[a] 表示工序 a 的加工时长 np.array
# mchMat 工序 - 设备映射矩阵（一维 / 二维），mchMat[a] 表示工序 a 对应的设备编号 np.array
# mchsStartTimes 各设备的任务起始时间列表，mchsStartTimes[m] 是设备 m 的所有任务起始时间 list[np.array]
# opIDsOnMchs 各设备的已分配工序列表，opIDsOnMchs[m] 是设备 m 已安排的工序编号 list[np.array]
def permissibleLeftShift(a, durMat, mchMat, mchsStartTimes, opIDsOnMchs):
    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs)
    dur_a = np.take(durMat, a)
    mch_a = np.take(mchMat, a) - 1
    startTimesForMchOfa = mchsStartTimes[mch_a]
    opsIDsForMchOfa = opIDsOnMchs[mch_a]
    flag = False

    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]
    # print('possiblePos:', possiblePos)
    if len(possiblePos) == 0:
        startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa)
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
        else:
            flag = True
            startTime_a = putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa)
    return startTime_a, flag


def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')
    index = np.where(startTimesForMchOfa == -1)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)
    startTimesForMchOfa[index] = startTime_a
    opsIDsForMchOfa[index] = a
    return startTime_a


def calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]
    durOfPossiblePos = np.take(durMat, opsIDsForMchOfa[possiblePos])
    startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0]-1] + np.take(durMat, [opsIDsForMchOfa[possiblePos[0]-1]]))
    endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care
    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
    idxLegalPos = np.where(dur_a <= possibleGaps)[0]
    legalPos = np.take(possiblePos, idxLegalPos)
    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]
    return startTime_a


def calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs):
    mch_a = np.take(mchMat, a) - 1
    # cal jobRdyTime_a
    jobPredecessor = a - 1 if a % mchMat.shape[1] != 0 else None
    if jobPredecessor is not None:
        durJobPredecessor = np.take(durMat, jobPredecessor)
        mchJobPredecessor = np.take(mchMat, jobPredecessor) - 1
        jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)] + durJobPredecessor).item()
    else:
        jobRdyTime_a = 0
    # cal mchRdyTime_a
    mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0 else None
    if mchPredecessor is not None:
        durMchPredecessor = np.take(durMat, mchPredecessor)
        mchRdyTime_a = (mchsStartTimes[mch_a][np.where(mchsStartTimes[mch_a] >= 0)][-1] + durMchPredecessor).item()
    else:
        mchRdyTime_a = 0

    return jobRdyTime_a, mchRdyTime_a


