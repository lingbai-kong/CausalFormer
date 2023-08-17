# This code snippet is sourced from the TCDF project by M-Nauta.
# Original code: https://github.com/M-Nauta/TCDF
# Licensed under the GNU General Public License v3.0 (GPL-3.0)
# SPDX-License-Identifier: GPL-3.0

import pandas as pd
import numpy as np
import networkx as nx
import copy

def getextendeddelays(gtfile, columns):
    """Collects the total delay of indirect causal relationships."""
    gtdata = pd.read_csv(gtfile, header=None)

    readgt=dict()
    effects = gtdata[1]
    causes = gtdata[0]
    delays = gtdata[2]
    gtnrrelations = 0
    pairdelays = dict()
    for k in range(len(columns)):
        readgt[k]=[]
    for i in range(len(effects)):
        key=effects[i]
        value=causes[i]
        readgt[key].append(value)
        pairdelays[(key, value)]=delays[i]
        gtnrrelations+=1
    
    g = nx.DiGraph()
    g.add_nodes_from(readgt.keys())
    for e in readgt:
        cs = readgt[e]
        for c in cs:
            g.add_edge(c, e)

    extendedreadgt = copy.deepcopy(readgt)
    
    for c1 in range(len(columns)):
        for c2 in range(len(columns)):
            paths = list(nx.all_simple_paths(g, c1, c2, cutoff=2)) #indirect path max length 3, no cycles
            if len(paths)>0:
                for path in paths:
                    for p in path[:-1]:
                        if p not in extendedreadgt[path[-1]]:
                            extendedreadgt[path[-1]].append(p)
                            
    extendedgtdelays = dict()
    for effect in extendedreadgt:
        causes = extendedreadgt[effect]
        for cause in causes:
            if (effect, cause) in pairdelays:
                delay = pairdelays[(effect, cause)]
                extendedgtdelays[(effect, cause)]=[delay]
            else:
                #find extended delay
                paths = list(nx.all_simple_paths(g, cause, effect, cutoff=2)) #indirect path max length 3, no cycles
                extendedgtdelays[(effect, cause)]=[]
                for p in paths:
                    delay=0
                    for i in range(len(p)-1):
                        delay+=pairdelays[(p[i+1], p[i])]
                    extendedgtdelays[(effect, cause)].append(delay)
    return extendedgtdelays, readgt, extendedreadgt
def evaluate(logger, gtfile, validatedcauses, columns):
    """Evaluates the results of TCDF by comparing it to the ground truth graph, and calculating precision, recall and F1-score. F1'-score, precision' and recall' include indirect causal relationships."""
    extendedgtdelays, readgt, extendedreadgt = getextendeddelays(gtfile, columns)
    FP=0
    FPdirect=0
    TPdirect=0
    TP=0
    FN=0
    FPs = []
    FPsdirect = []
    TPsdirect = []
    TPs = []
    FNs = []
    for key in readgt:
        for v in validatedcauses[key]:
            if v not in extendedreadgt[key]:
                FP+=1
                FPs.append((key,v))
            else:
                TP+=1
                TPs.append((key,v))
            if v not in readgt[key]:
                FPdirect+=1
                FPsdirect.append((key,v))
            else:
                TPdirect+=1
                TPsdirect.append((key,v))
        for v in readgt[key]:
            if v not in validatedcauses[key]:
                FN+=1
                FNs.append((key, v))
    
    def serialization(data):
        return [f"{e[1]}->{e[0]}" for e in data]
    logger.info(f"Total False Positives': {FP}")
    logger.info(f"Total True Positives': {TP}")
    logger.info(f"Total False Negatives: {FN}")
    logger.info(f"Total Direct False Positives: {FPdirect}")
    logger.info(f"Total Direct True Positives: {TPdirect}")
    logger.info(f"TPs': {serialization(TPs)}")
    logger.info(f"FPs': {serialization(FPs)}")
    logger.info(f"TPs direct: {serialization(TPsdirect)}")
    logger.info(f"FPs direct: {serialization(FPsdirect)}")
    logger.info(f"FNs: {serialization(FNs)}")
    precision = recall = 0.

    logger.info('(includes direct and indirect causal relationships)')
    if float(TP+FP)>0:
        precision = TP / float(TP+FP)
    logger.info(f"Precision': {precision}")
    if float(TP + FN)>0:
        recall = TP / float(TP + FN)
    logger.info(f"Recall': {recall}")
    if (precision + recall) > 0:
        F1 = 2 * (precision * recall) / (precision + recall)
    else:
        F1 = 0.
    logger.info(f"F1' score: {F1}")

    logger.info('(includes only direct causal relationships)')
    precision = recall = 0.
    if float(TPdirect+FPdirect)>0:
        precision = TPdirect / float(TPdirect+FPdirect)
    logger.info(f"Precision: {precision}")
    if float(TPdirect + FN)>0:
        recall = TPdirect / float(TPdirect + FN)
    logger.info(f"Recall: {recall}")
    if (precision + recall) > 0:
        F1direct = 2 * (precision * recall) / (precision + recall)
    else:
        F1direct = 0.
    logger.info(f"F1 score: {F1direct}")
    return FP, TP, FPdirect, TPdirect, FN, FPs, FPsdirect, TPs, TPsdirect, FNs, F1, F1direct
def evaluatedelay(extendedgtdelays, alldelays, TPs, receptivefield):
    """Evaluates the delay discovery of TCDF by comparing the discovered time delays with the ground truth."""
    zeros = 0
    total = 0.
    for i in range(len(TPs)):
        tp=TPs[i]
        discovereddelay = alldelays[tp]
        gtdelays = extendedgtdelays[tp]
        for d in gtdelays:
            if d <= receptivefield:
                total+=1.
                error = d - discovereddelay
                if error == 0:
                    zeros+=1
            else:
                next
    if zeros==0:
        return 0.
    else:
        return zeros/float(total)