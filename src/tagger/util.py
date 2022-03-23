import torch
import torch.nn as nn
import json
import random
import pickle
import re

def get_doc_and_sum(sumdatapath, usetrain, usevalid):
    trainfile = sumdatapath + "/train.txt"
    validfile = sumdatapath + "/valid.txt"
    alldoc = []
    allsum = []
    if usetrain:
        f = open(trainfile,'r')
        while True:
            oneline = f.readline().strip()
            if not oneline:
                break
            onedata = oneline.split('\t')
            if len(onedata) != 2:
                continue
            #onedoc = onedata[0]
            onedoc = re.sub(' +', ' ', onedata[0])
            #onesum = onedata[1]
            onesum = re.sub(' +', ' ', onedata[1])
            alldoc.append(onedoc)
            allsum.append(onesum)
        f.close()
    if usevalid:
        f = open(validfile, 'r')
        while True:
            oneline = f.readline().strip()
            if not oneline:
                break
            onedata = oneline.split('\t')
            if len(onedata) != 2:
                continue
            #onedoc = onedata[0]
            onedoc = re.sub(' +', ' ', onedata[0])
            #onesum = onedata[1]
            onesum = re.sub(' +', ' ', onedata[1])
            alldoc.append(onedoc)
            allsum.append(onesum)
        f.close()
    print(len(alldoc))
    print(len(allsum))
    docpath = sumdatapath + "/doc.txt"
    sumpath = sumdatapath + "/sum.txt"
    f = open(docpath,'w')
    for oned in alldoc:
        f.write(oned+"\n")
    f.close()
    f = open(sumpath,'w')
    for ones in allsum:
        f.write(ones+"\n")
    f.close()
    return docpath,sumpath


def getfilewithlabel(file, filewithfakelabel):

    fin = open(file,'r')
    alldata = []
    while True:
        oneline = fin.readline().strip()
        if not oneline:
            break
        #oneline = re.sub(' +', ' ', oneline)
        alldata.append(oneline.split(' '))
    fin.close()

    #print(len(alldata))

    fo = open(filewithfakelabel, 'w')
    fo.write("-DOCSTART- -X- -X- O\n")
    fo.write("\n")
    for onedata in alldata:
        datasize = len(onedata)
        for i in range(datasize):
            fo.write(onedata[i]+" NNP B-NP O\n")
        fo.write("\n")
    fo.close()
    return alldata

# def getdocandent(docfile,allentitylist,alltypelist):
#     f = open(docfile,'r')
#     alldoc = []
#     while True:
#         oneline = f.readline().strip()
#         if not oneline:
#             break
#         alldoc.append(oneline)
#     f.close()
#     allres = []
#     resfortrain = []
#     trainsize = len(alldoc) // 2
#     assert len(alldoc) == len(allentitylist)
#     for i in range(len(alldoc)):
#         ######split to shorter sentences
#         onedoclist = alldoc[i].split(' ')
#         if i < trainsize:
#             resfortrain.append(
#                 ' '.join(onedoclist) + "\t" + '!'.join(allentitylist[i]) + "\t" + '?'.join(alltypelist[i]))
#         num = 100
#         newlist = []
#         for j in range(0, len(onedoclist), num):
#             newlist.append(onedoclist[j:j + num])
#             # print(len(onedoclist[j:j+num]))
#
#         for j in range(len(newlist)):
#             allres.append(' '.join(newlist[j]) + "\t" + '!'.join(allentitylist[i]) + "\t" + '?'.join(alltypelist[i]))
#     return allres, resfortrain

def getdocandent(docfile,allentitylist,alltypelist):
    f = open(docfile,'r')
    alldoc = []
    while True:
        oneline = f.readline().strip()
        if not oneline:
            break
        alldoc.append(oneline)
    f.close()

    allrestrain = []
    allresvalid = []

    resfortrain = []
    trainsize = len(alldoc) // 2
    assert len(alldoc) == len(allentitylist)
    for i in range(len(alldoc)):
        ######split to shorter sentences
        onedoclist = alldoc[i].split(' ')
        if i < trainsize:
            resfortrain.append(' '.join(onedoclist) + "\t" + '!'.join(allentitylist[i]) + "\t" + '?'.join(alltypelist[i]))
        num = 100
        newlist = []
        for j in range(0, len(onedoclist), num):
            newlist.append(onedoclist[j:j+num])
            #print(len(onedoclist[j:j+num]))
        if i < len(alldoc) // 2:
            for j in range(len(newlist)):
                allrestrain.append(
                    ' '.join(newlist[j]) + "\t" + '!'.join(allentitylist[i]) + "\t" + '?'.join(alltypelist[i]))
        else:
            for j in range(len(newlist)):
                allresvalid.append(
                    ' '.join(newlist[j]) + "\t" + '!'.join(allentitylist[i]) + "\t" + '?'.join(alltypelist[i]))
    return allrestrain, allresvalid, resfortrain

def getindex(str1,str2):
    ###if str1 in str2
    str1list = str1.lower().split(' ')
    str2list = str2.lower().split(' ')
    ifin = False
    allindex = []
    for i in range(0,len(str2list)):
        if str2list[i] == str1list[0]:
            ifin = True
            oneindex = [i]
            for j in range(1,len(str1list)):
                if i+j >= len(str2list) or str2list[i+j] != str1list[j]:
                    ifin = False
                    break
                else:
                    oneindex.append(i+j)
            if ifin == True:
                allindex.append(oneindex)
        else:
            continue
    return allindex