import numpy as np

import torch
from torch import nn
from pytorch_pretrained_bert import BertModel

from misc import flat_list
from misc import sim_matrix, contrastive_loss
from misc import iterative_support, conflict_judge
from utils import UnitAlphabet, LabelAlphabet
from sklearn.metrics.pairwise import cosine_similarity


class PhraseClassifier(nn.Module):

    def __init__(self,
                 lexical_vocab: UnitAlphabet,
                 label_vocab: LabelAlphabet,
                 hidden_dim: int,
                 dropout_rate: float,
                 neg_rate: float,
                 clloss_percent: float,
                 score_percent: float,
                 cl_scale: int,
                 cl_temp: float,
                 use_detach: bool,
                 bert_path: str):
        super(PhraseClassifier, self).__init__()

        self._lexical_vocab = lexical_vocab
        self._label_vocab = label_vocab
        self._neg_rate = neg_rate
        self._clloss_percent = clloss_percent
        self._score_percent = score_percent
        self._cl_scale = cl_scale
        self._cl_temp = cl_temp
        self._use_detach = use_detach
        self._encoder = BERT(bert_path)
        self._classifier = MLP(self._encoder.dimension * 4, hidden_dim, len(label_vocab), dropout_rate)
        self._criterion = nn.NLLLoss()

    def forward(self, var_h, **kwargs):
        con_repr = self._encoder(var_h, kwargs["mask_mat"], kwargs["starts"])

        batch_size, token_num, hidden_dim = con_repr.size()
        ext_row = con_repr.unsqueeze(2).expand(batch_size, token_num, token_num, hidden_dim)
        ext_column = con_repr.unsqueeze(1).expand_as(ext_row)
        table = torch.cat([ext_row, ext_column, ext_row - ext_column, ext_row * ext_column], dim=-1)
        return self._classifier(table), self._classifier.get_dense(table)

    def _pre_process_input(self, utterances):
        lengths = [len(s) for s in utterances]
        max_len = max(lengths)
        pieces = iterative_support(self._lexical_vocab.tokenize, utterances)
        units, positions = [], []

        for tokens in pieces:
            units.append(flat_list(tokens))
            cum_list = np.cumsum([len(p) for p in tokens]).tolist()
            positions.append([0] + cum_list[:-1])

        sizes = [len(u) for u in units]
        max_size = max(sizes)
        cls_sign = self._lexical_vocab.CLS_SIGN
        sep_sign = self._lexical_vocab.SEP_SIGN
        pad_sign = self._lexical_vocab.PAD_SIGN
        pad_unit = [[cls_sign] + s + [sep_sign] + [pad_sign] * (max_size - len(s)) for s in units]
        starts = [[ln + 1 for ln in u] + [max_size + 1] * (max_len - len(u)) for u in positions]

        var_unit = torch.LongTensor([self._lexical_vocab.index(u) for u in pad_unit])
        attn_mask = torch.LongTensor([[1] * (lg + 2) + [0] * (max_size - lg) for lg in sizes])
        var_start = torch.LongTensor(starts)

        if torch.cuda.is_available():
            var_unit = var_unit.cuda()
            attn_mask = attn_mask.cuda()
            var_start = var_start.cuda()
        return var_unit, attn_mask, var_start, lengths

    def _pre_process_output(self, entities, lengths):
        positions, labels = [], []
        batch_size = len(entities)

        for utt_i in range(0, batch_size):
            for segment in entities[utt_i]:
                positions.append((utt_i, segment[0], segment[1]))
                labels.append(segment[2])

        for utt_i in range(0, batch_size):
            reject_set = [(e[0], e[1]) for e in entities[utt_i]]
            s_len = lengths[utt_i]
            neg_num = int(s_len * self._neg_rate) + 1

            candies = flat_list([[(i, j) for j in range(i, s_len) if (i, j) not in reject_set] for i in range(s_len)])
            if len(candies) > 0:
                sample_num = min(neg_num, len(candies))
                assert sample_num > 0

                np.random.shuffle(candies)
                for i, j in candies[:sample_num]:
                    positions.append((utt_i, i, j))
                    labels.append("O")

        var_lbl = torch.LongTensor(iterative_support(self._label_vocab.index, labels))
        if torch.cuda.is_available():
            var_lbl = var_lbl.cuda()
        return positions, var_lbl


    def estimate_CL(self, sentences, segments):
        var_sent, attn_mask, start_mat, lengths = self._pre_process_input(sentences)
        score_t, embedding_t = self(var_sent, mask_mat=attn_mask, starts=start_mat)

        positions, targets = self._pre_process_output(segments, lengths)
        targets = targets.cuda()
        flat_s = torch.cat([score_t[[i], j, k] for i, j, k in positions], dim=0).cuda()
        flat_e = torch.cat([embedding_t[[i], j, k] for i, j, k in positions], dim=0).cuda()
        softmax_score = torch.log_softmax(flat_s, dim=-1)
        CE_loss = self._criterion(softmax_score, targets.cuda())
        CL_loss = contrastive_loss(flat_e.cuda(), targets.cuda(), detach = self._use_detach,temp = self._cl_temp, scale = self._cl_scale)


        dict_center = {}
        dict_num = {}
        target_num = softmax_score.size()[1]
        
        for i in range(0,target_num):
            dict_center[i] = torch.zeros(embedding_t.size()[3])
            dict_num[i] = 0
        for i in range(0,len(targets)):
            dict_num[targets[i].item()] = dict_num[targets[i].item()] + 1
        for i in range(0,len(flat_e)):
            dict_center[targets[i].item()] = dict_center[targets[i].item()] + (flat_e[i].detach().cpu()/dict_num[targets[i].item()])

        return self._clloss_percent * CL_loss + (1-self._clloss_percent) * CE_loss, dict_center

    def inference(self, sentences, dict_center):
        var_sent, attn_mask, starts, lengths = self._pre_process_input(sentences)
        log_items, embedding_t = self(var_sent, mask_mat=attn_mask, starts=starts)
        score_t = torch.log_softmax(log_items, dim=-1) 


        bz, len_1 ,len_2 ,hidden_len = embedding_t.size()
        embedding_t = embedding_t.view(-1,hidden_len)
        center_tensor = torch.stack(list(dict_center.values()))
        
        
        distance_score = sim_matrix(embedding_t.cpu(), center_tensor.cpu())
        distance_score = torch.softmax(distance_score, dim=-1)
        distance_score[:,0] = 0 
        distance_score = distance_score.view(bz,len_1,len_2,-1)

        
        score_result = (1-self._score_percent) * score_t.cpu() + self._score_percent * distance_score.cpu()
        val_table, idx_table = torch.max(score_result, dim=-1)

        listing_it = idx_table.cpu().numpy().tolist()
        listing_vt = val_table.cpu().numpy().tolist()
        label_table = iterative_support(self._label_vocab.get, listing_it)

        candidates = []
        for l_mat, v_mat, sent_l in zip(label_table, listing_vt, lengths):
            candidates.append([])
            for i in range(0, sent_l):
                for j in range(i, sent_l):
                    if l_mat[i][j] != "O":
                        candidates[-1].append((i, j, l_mat[i][j], v_mat[i][j]))

        entities = []
        for segments in candidates:
            ordered_seg = sorted(segments, key=lambda e: -e[-1])
            filter_list = []
            for elem in ordered_seg:
                flag = False
                current = (elem[0], elem[1])
                for prior in filter_list:
                    flag = conflict_judge(current, (prior[0], prior[1]))
                    if flag:
                        break
                if not flag:
                    filter_list.append((elem[0], elem[1], elem[2]))
            entities.append(sorted(filter_list, key=lambda e: e[0]))
        return entities


class BERT(nn.Module):

    def __init__(self, source_path):
        super(BERT, self).__init__()
        self._repr_model = BertModel.from_pretrained(source_path)

    @property
    def dimension(self):
        return 768

    def forward(self, var_h, attn_mask, starts):
        all_hidden, _ = self._repr_model(var_h, attention_mask=attn_mask, output_all_encoded_layers=False)

        batch_size, _, hidden_dim = all_hidden.size()
        _, unit_num = starts.size()
        positions = starts.unsqueeze(-1).expand(batch_size, unit_num, hidden_dim)
        return torch.gather(all_hidden, dim=-2, index=positions)


class MLP(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate):
        super(MLP, self).__init__()

        self._densenet = nn.Sequential(nn.Linear(input_dim,hidden_dim),
                                      nn.Tanh())
        self._scorer = nn.Linear(hidden_dim, output_dim)
        self._dropout = nn.Dropout(dropout_rate)


    def forward(self, var_h):
        return self._scorer(self._densenet(self._dropout(var_h)))

    def get_dense(self, var_h):
        return self._densenet(self._dropout(var_h))

    def get_score(self, dense):
        return self._scorer(dense)
