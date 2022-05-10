import time
import json
import codecs
import os
import numpy as np
import random

import torch


def iterative_support(func, query):
    if isinstance(query, (list, tuple, set)):
        return [iterative_support(func, i) for i in query]
    return func(query)

def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """

    m, n = x.size(0), y.size(0)
    # xx经过pow()方法对每单个数据进行二次方操作后，在axis=1 方向（横向，就是第一列向最后一列的方向）加和，此时xx的shape为(m, 1)，经过expand()方法，扩展n-1次，此时xx的shape为(m, n)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    # yy会在最后进行转置的操作
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    # torch.addmm(beta=1, input, alpha=1, mat1, mat2, out=None)，这行表示的意思是dist - 2 * x * yT
    dist.addmm_(1, -2, x, y.t())
    # clamp()函数可以限定dist内元素的最大最小范围，dist最后开方，得到样本之间的距离矩阵
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return 1/dist

def fix_random_seed(state_val):
    random.seed(state_val)
    np.random.seed(state_val)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(state_val)
        torch.cuda.manual_seed_all(state_val)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    torch.manual_seed(state_val)
    torch.random.manual_seed(state_val)


def flat_list(h_list):
    e_list = []

    for item in h_list:
        if isinstance(item, list):
            e_list.extend(flat_list(item))
        else:
            e_list.append(item)
    return e_list


def f1_score(sent_list, pred_list, gold_list, script_path):
    fn_out = 'eval_%04d.txt' % random.randint(0, 10000)
    if os.path.isfile(fn_out):
        os.remove(fn_out)

    text_file = open(fn_out, mode='w', encoding='utf-8')
    for i, words in enumerate(sent_list):
        tags_1 = gold_list[i]
        tags_2 = pred_list[i]
        for j, word in enumerate(words):
            tag_1 = tags_1[j]
            tag_2 = tags_2[j]
            text_file.write('%s\t%s\t%s\n' % (word, tag_1, tag_2))
        text_file.write('\n')
    text_file.close()

    cmd = 'perl %s -d "\t" < %s' % (script_path, fn_out)
    msg = '\nStandard CoNNL perl script (author: Erik Tjong Kim Sang <erikt@uia.ua.ac.be>, version: 2004-01-26):\n'
    msg += ''.join(os.popen(cmd).readlines())
    time.sleep(1.0)
    if fn_out.startswith('eval_') and os.path.exists(fn_out):
        os.remove(fn_out)
    #print(msg)
    return float(msg.split('\n')[3].split(':')[-1].strip())


def iob_tagging(entities, s_len):
    tags = ["O"] * s_len

    for el, er, et in entities:
        for i in range(el, er + 1):
            if i == el:
                tags[i] = "B-" + et
            else:
                tags[i] = "I-" + et
    return tags


def conflict_judge(line_x, line_y):
    if line_x[0] == line_y[0]:
        return True
    if line_x[0] < line_y[0]:
        if line_x[1] >= line_y[0]:
            return True
    if line_x[0] > line_y[0]:
        if line_x[0] <= line_y[1]:
            return True
    return False


def extract_json_data(file_path):
    with codecs.open(file_path, "r", "utf-8") as fr:
        dataset = json.load(fr)
    return dataset

def sim_matrix(a, b, eps=1e-8):
    a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
    a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
    b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))
    sim_mt = torch.mm(a_norm, b_norm.transpose(0, 1))
    return sim_mt

def contrastive_loss(embedding, label, detach = False, temp=0.1, scale=100):
    if detach:
        cosine_sim = sim_matrix(embedding,embedding).detach() 
        dis = cosine_sim[~torch.eye(cosine_sim.shape[0], dtype=torch.bool)].reshape(cosine_sim.shape[0], -1).detach()
    else:
        cosine_sim = sim_matrix(embedding, embedding)
        dis = cosine_sim[~torch.eye(cosine_sim.shape[0], dtype=torch.bool)].reshape(cosine_sim.shape[0], -1)
    
    
    dis = dis / temp
    cosine_sim = cosine_sim / temp
    
    dis = torch.exp(dis)
    cosine_sim = torch.exp(cosine_sim)

    
    row_sum = []
    for i in range(len(embedding)):
        row_sum.append(sum(dis[i]))
    
    contrastive_loss = 0
    for i in range(len(embedding)):
        n_i = label.tolist().count(label[i]) - 1
        inner_sum = 0
        for j in range(len(embedding)):
            if label[i] == label[j] and i != j:
                inner_sum = inner_sum + torch.log(cosine_sim[i][j] / row_sum[i])
        if n_i != 0:
            contrastive_loss += (inner_sum / (-n_i))
        else:
            contrastive_loss += 0
    return contrastive_loss / scale

