import sys
import re
import os
import logging as log


class Stack:
    def __init__(self):
        self.items = []

    def isEmpty(self):
        return len(self.items) == 0

    def push(self, item):
        self.items.append(item)

    def pop(self):
        return self.items.pop()

    def peek(self):
        if not self.isEmpty():
            return self.items[len(self.items) - 1]

    def size(self):
        return len(self.items)

class Graph:
    def __init__(self, maps = {}):
        self.map = maps       #
        self.nodenum = len(maps)
        self.nodes = list(range(self.nodenum))

    def find_path(self, start, end, path=[]):
        path = path + [start]
        if start == end:
            return path
        if start not in self.map:
            return None
        for node in self.map[start]:
            if node not in path:
                newpath = self.find_path(node, end, path)
                if newpath: return newpath
        return None

def list_to_file(lst, out_file):
    '''Write list to file'''
    with open(out_file, "w") as out_f:
        for line in lst:
            out_f.write(line.strip() + '\n')
    out_f.close()

def file_to_list(in_file):
    '''Write list to file'''
    lst = []
    with open(in_file, "r") as in_f:
        for line in in_f:
            lst.append(line.strip())
    return lst


def get_sbn(f):
    '''Read and return individual sbns in clause format'''
    cur_sbn = []
    all_sbns = []
    for line in open(f, 'r'):
        annotaion = re.search('%%%', line)
        if not line.strip():
            if cur_sbn:
                all_sbns.append(cur_sbn)
                cur_sbn = []
        else:
            if annotaion:
                # delete first three lines comments
                continue
            else:
                cur_sbn.append(line.strip())
    ## If we do not end with a newline we should add the sbn still
    if cur_sbn:
        all_sbns.append(cur_sbn)
    return all_sbns

def sbn_string_to_list(sbn):
    '''Change a sbn in string format (single list) to a list of lists
       Also remove comments from the sbn'''
    sbn = [x for x in sbn if x.strip() and not x.startswith('%')]
    sbn = [x.split('%')[0].strip() for x in sbn]
    sbn = [clause.split()[0:clause.split().index('%')] if '%' in clause.split() else clause.split() for clause in sbn]
    return sbn

def get_dir_sen(inputpath, sen_dir, outfile):
    with open(outfile,'w') as out_f:
        for line in sen_dir:
            raw_path = os.path.join(inputpath, "{}/en.raw".format(line.strip()))
            with open(raw_path, 'r') as raw_f:
                size = os.path.getsize(raw_path)
                if size != 0:
                    a = ''
                    sentences = raw_f.readlines()
                    for sentence in sentences:
                        a += sentence.strip()+' '
                    out_f.writelines(a + '\n')
                else:
                    log.info("The raw sentence file {} does not exist".format(raw_path))
    return out_f

def get_dir_sbn(inputpath, sbn_dir, outfile):
    with open(outfile,'w') as out_f:
        for line in sbn_dir:
            sbn_path = os.path.join(inputpath, "{}/en.drs.sbn".format(line.strip()))
            with open(sbn_path, 'r') as sbn_f:
                content = sbn_f.readlines()
                out_f.writelines(content)
                out_f.writelines('\n')



def is_number(n):
    try:
        num = float(n)
        # 检查 "nan"
        is_number = num == num  # 或者使用 `math.isnan(num)`
    except ValueError:
        is_number = False
    return is_number

def between_quotes(string):
    '''Return true if a value is between quotes'''
    return (string.startswith('"') and string.endswith('"')) or (string.startswith("'") and string.endswith("'"))

def include_quotes(string):
    '''Return true if a value is between quotes'''
    return string.startswith('"') or string.endswith('"') or string.startswith("'") or string.endswith("'")

def is_operator(string):
    '''Checks if all items in a string are uppercase'''
    return all(x.isupper() or x.isdigit() for x in string) and string[0].isupper()

def is_role(string):
    '''Check if string is in the format of a role'''
    return string[0].isupper() and any(x.islower() for x in string[1:]) and all(x.islower() or x.isupper() or x == '-' for x in string)

### Name (the quantity after Name) and Quantity need to preprocess ###
# cur_sbn is current SBN piece list, adn tuple is needed to add #
def add_tuple(simple_sbn, id_cur_sbn, cur_sbn, tuple_sbn, nodes_sbn, index):
    for j in range(index, len(cur_sbn) - 2, 2):
        # if "Quantity" in cur_sbn[j + 1]:
        #     nodes_sbn.append(cur_sbn[j + 2])
        #     tuple_sbn.append([cur_sbn[0], cur_sbn[j + 1], cur_sbn[j + 2]])
        if is_number(cur_sbn[j + 2]) and (
                re.search("\+", cur_sbn[j + 2]) or re.search("\-", cur_sbn[j + 2])):
            try:
                tuple_sbn.append([cur_sbn[0], cur_sbn[j + 1], simple_sbn[id_cur_sbn + int(cur_sbn[j + 2])][0]])
            except:  # p04/d2291; p13/d1816
                nodes_sbn.append(cur_sbn[j + 2])
                tuple_sbn.append([cur_sbn[0], cur_sbn[j + 1], cur_sbn[j + 2]])
        else:
            nodes_sbn.append(cur_sbn[j + 2])
            tuple_sbn.append([cur_sbn[0], cur_sbn[j + 1], cur_sbn[j + 2]])
    return tuple_sbn, nodes_sbn

def add_tuple_box(sbn, tuple_sbn, nodes_sbn, TOPIC):
    ## This sbn include BOS and Negation information
    Relation = [x for x in sbn if x[0].isupper() == True]
    temp = '' 
    for i, cur_sbn in enumerate(sbn):
        if len(cur_sbn) == 2:
            continue
        else:
            for j, x in enumerate(cur_sbn):
                if 'BOS' in x:
                    temp = x
                    nodes_sbn.append(x)
                else:
                    if x.strip()==TOPIC.strip(): 
                        tuple_sbn.append([temp, "TOPIC", x])
                    elif j==0:
                        tuple_sbn.append([temp, ":IMP", x])
    # add tuple about BOS and NEGATION
    for i, item in enumerate(Relation):
        if len(Relation[i]) == 2:
            assert i != 0
            tuple_sbn.append([Relation[i - 1][0], Relation[i][0], Relation[i + 1][0]])
    return tuple_sbn, nodes_sbn

def get_concept(redundant_line):
    concept = []
    for i, cur_sbn in enumerate(redundant_line):
        if len(cur_sbn) == 1:
            concept.extend([cur_sbn[0]])
        elif len(cur_sbn) == 2:
            continue
        else:
            concept.extend([cur_sbn[0]])
            for j in range(0, len(cur_sbn) - 2, 2):
                # if "Quantity" in cur_sbn[j + 1]:
                #     concept.extend([cur_sbn[j+2]])
                if is_number(cur_sbn[j + 2]) and not re.search("\+", cur_sbn[j + 2]) and not re.search("\-", cur_sbn[j + 2]):
                    concept.extend([cur_sbn[j + 2]])
                elif is_number(cur_sbn[j + 2]) and 767:
                    try:
                        redundant_line[i + int(cur_sbn[j + 2])][0]
                    except:
                        concept.extend([cur_sbn[j + 2]])
                else:
                    concept.extend([cur_sbn[j + 2]])
    return concept

def get_topic(simple_sbn, id_cur_sbn, cur_sbn, index):
    TOPIC=''
    verb_role_list = ['Agent', 'Patient']
    for j in range(index, len(cur_sbn) - 2, 2):
        if cur_sbn[j + 1] in verb_role_list and is_number(cur_sbn[j + 2]) and re.search("\+", cur_sbn[j + 2]):
            try:
                TOPIC= simple_sbn[id_cur_sbn + int(cur_sbn[j + 2])][0].strip()
            except:
                continue
    return TOPIC

def get_tuple(line):
    tuple_sbn = []
    nodes_sbn = []
    # -------------------------------------------------------------------
    redundant_line = []  # 冗余的序列，增加了Box信息
    box_number = 0
    for cur_sbn in line:
        if len(redundant_line) == 0:
            box_number += 1
            redundant_line.append(['BOS' + str(box_number)])
        if len(cur_sbn) == 2:
            redundant_line.append(cur_sbn)
            box_number += 1
            redundant_line.append(['BOS' + str(box_number)])
        else:
            redundant_line.append(cur_sbn)
    # --------------------------------------------------------------------
    vocab_list = []# 如果列表中的元素重复则更新，使其每个元素都独一无二
    new_redundant_line = []
    for i, cur_sbn in enumerate(redundant_line):
        new_cur_sbn = cur_sbn
        for j, item in enumerate(cur_sbn):
            if is_number(item) and (re.search("\+", item) or re.search("\-", item)):
                continue
            elif item not in vocab_list:
                vocab_list.append(item)
            elif item in vocab_list:
                new_cur_sbn[j] = cur_sbn[j] + '^' * (i+j)
        new_redundant_line.append(new_cur_sbn)
    # ---------------------------------------------------------------
    # 简化的数据: 没有BOS 和 Negation 信息 # 得到所有的三元组
    simple_line = [x for x in new_redundant_line if (x[0].isupper() == False) and ('BOS' not in x[0]) ] # 简化的序列，没有Negation 信息
    if_TOPIC=''
    for i, cur_sbn in enumerate(simple_line):
        nodes_sbn.append(cur_sbn[0])
        if len(cur_sbn) == 1:
            continue
        elif ("Name" or "EQU") in cur_sbn[1] and include_quotes(cur_sbn[2]):
            # a,b = [],[]
            for nn in range(2, len(cur_sbn), 1):
                if cur_sbn[nn].endswith('"') or cur_sbn[nn].strip('^').endswith('"'): 
                    end_index = nn
                    for index in range(2, end_index+1):
                        nodes_sbn.append(cur_sbn[index])
                        tuple_sbn.append([cur_sbn[0], cur_sbn[1], cur_sbn[index]])
                    break
            ### 超过name以后的role relation ## For special case: /p43/d2796
            if end_index + 2 < len(cur_sbn):
                if ("Name" or "EQU") in cur_sbn[end_index + 1] and include_quotes(cur_sbn[end_index + 2]):
                    new_index = end_index + 2
                    for nn in range(new_index, len(cur_sbn), 1):
                        if cur_sbn[nn].endswith('"') or cur_sbn[nn].strip('^').endswith('"'):
                            end_index = nn
                            for index in range(new_index, end_index + 1):
                                nodes_sbn.append(cur_sbn[index])
                                tuple_sbn.append([cur_sbn[0], 'Name', cur_sbn[index]])
            ###超过name以后的role relation
            if end_index + 2 < len(cur_sbn):
                tuple_sbn, nodes_sbn = add_tuple(simple_line, i, cur_sbn, tuple_sbn, nodes_sbn, index=end_index)
        else:
            tuple_sbn, nodes_sbn = add_tuple(simple_line, i, cur_sbn, tuple_sbn, nodes_sbn,index=0)
        role_list = ['Time', 'Agent', 'Patient']
        if set(role_list) < set(cur_sbn):
            if_TOPIC = get_topic(simple_line, i, cur_sbn, index=0) 
    # add tuple about BOS and :IMP
    tuple_sbn, nodes_sbn = add_tuple_box(new_redundant_line, tuple_sbn, nodes_sbn, if_TOPIC)
    return tuple_sbn, nodes_sbn


# delete Name ""
def word_level_sbn(new_clauses):
    '''Return to string format, use char-level for concepts'''
    return_strings = []
    for cur_clause in new_clauses:
        for idx, item in enumerate(cur_clause):
            if cur_clause[idx-1] == 'Name' and between_quotes(cur_clause[idx]):
                cur_clause[idx] = item.strip('"')
            if cur_clause[idx].startswith('"') and not cur_clause[idx].endswith('"'):
                for j in range(idx, len(cur_clause), 1):
                    if cur_clause[j].endswith('"'):
                        end_index = j
                        for index in range(idx, end_index+1):
                            cur_clause[index] = cur_clause[index].strip('"')
        return_strings.append(cur_clause)
    return return_strings
def char_level_sbn(new_clauses):
    '''Return to string format, use char-level for concepts'''
    return_strings = []
    for cur_clause in new_clauses:
        for idx, item in enumerate(cur_clause):
            if cur_clause[idx-1] == 'Name' and between_quotes(cur_clause[idx]):
                cur_clause[idx] = " ".join(item.strip('"'))
            if cur_clause[idx-1] == 'Quantity':
                cur_clause[idx] = " ".join(item.strip(''))
            if cur_clause[idx].startswith('"') and not cur_clause[idx].endswith('"'):
                for j in range(idx, len(cur_clause), 1):
                    if cur_clause[j].endswith('"'):
                        end_index = j
                        for index in range(idx, end_index+1):
                            cur_clause[index] = " ".join(cur_clause[index].strip('"'))
        return_strings.extend(cur_clause)
    return return_strings

def read_anonymized(concept, tuple):
    # get concept dictionary
    idxmap = {}
    for i, item in enumerate(concept):
        idxmap[item] = i
    sbn_edges = []
    for i, item in enumerate(tuple):
        edge_lable = item[1]
        sbn_edges.append((idxmap[item[0]],idxmap[item[2]], edge_lable.strip('^')))
    clean_concept = [x.strip('"') for x in concept]
    return clean_concept, sbn_edges

def read_sbn_file(sbn_path):
    nodes = []
    in_neigh_indices, in_neigh_edges, out_neigh_indices, out_neigh_edges = [],[],[],[]
    max_in_neigh, max_out_neigh, max_node = 0, 0, 0
    sbns = get_sbn(sbn_path)
    for sbn in sbns:
        ### get sequence sbn data from raw sbn data
        seq_sbn = sbn_string_to_list(sbn)
        ### get all the tuples of sbn
        tuple, concept = get_tuple(seq_sbn)
        ### get edges format like: [(0, 1, ':IMP'), (2, 3, 'EQU'), (2, 4, 'Instrument'), (0, 2, 'AttributeOf')]
        sbn_node, sbn_edge = read_anonymized(concept, tuple)
        # 1.
        clean_concept = [x.strip('^').strip('"') for x in sbn_node]
        nodes.append(clean_concept)
        # 2. & 3.
        in_indices = [[i, ] for i, x in enumerate(sbn_node)]
        in_edges = [[':self', ] for i, x in enumerate(sbn_node)]
        out_indices = [[i, ] for i, x in enumerate(sbn_node)]
        out_edges = [[':self', ] for i, x in enumerate(sbn_node)]
        for (i, j, lb) in sbn_edge:
            in_indices[j].append(i)
            in_edges[j].append(lb)
            out_indices[i].append(j)
            out_edges[i].append(lb)
        in_neigh_indices.append(in_indices)
        in_neigh_edges.append(in_edges)
        out_neigh_indices.append(out_indices)
        out_neigh_edges.append(out_edges)
        # update lengths
        max_in_neigh = max(max_in_neigh, max(len(x) for x in in_indices))
        max_out_neigh = max(max_out_neigh, max(len(x) for x in out_indices))
        max_node = max(max_node, len(sbn_node))
    return zip(nodes, in_neigh_indices, in_neigh_edges, out_neigh_indices, out_neigh_edges), \
           max_node, max_in_neigh, max_out_neigh


def apply_bpe_edge(bpe_node_lst, node_lst, G1, tag1, G2, tag2):
    bpe2word = {}
    word2bpe = {}
    bpe_lst = bpe_node_lst
    start, index = 0, 0
    for i, word in enumerate(node_lst):
        word2bpe[i] = []
        try:
            if bpe_lst[start].endswith('@@'):
                pass
        except:
            print(node_lst)
            print(bpe_node_lst)
            exit()
        if bpe_lst[start].endswith('@@'):
            index = start + 1
            while bpe_lst[index].endswith('@@'):
                index += 1
            cat_bpe = (''.join(bpe_lst[start:index + 1])).replace('@@', '')
            assert word == cat_bpe, 'Inconsistent bpe {}->{}'.format(cat_bpe, word)
            for j in range(start, index):
                bpe2word[j] = (i, False)
                word2bpe[i].append(j)
            bpe2word[index] = (i, True)  # Last bpe part
            word2bpe[i].append(index)
            start = index + 1
        else:
            assert word == bpe_lst[start], 'Inconsistent bpe {}->{}'.format(bpe_lst[start], word)
            bpe2word[start] = (i, True)
            word2bpe[i].append(start)
            start += 1

    ############transforme G1#############
    G1_new, tag1_new = [], []
    for i in range(len(bpe_node_lst)):
        Gi, tagi = [], []
        i_parent = bpe2word[i][0]
        for Node_index, tag in zip(G1[i_parent], tag1[i_parent]):
            if Node_index == i_parent:  # Node to itself
                Gi.append(i)
                assert tag == ':self', 'Inconsistent tag, should be :self but get {}'.format(tag)
                tagi.append(':self')
            else:  # Other node to node: i_parent
                Gi.append(word2bpe[Node_index][-1])  # last children of Node_index
                tagi.append(tag)  # same tag

        G1_new.append(Gi)
        tag1_new.append(tagi)

    ############transforme G2#############
    G2_new, tag2_new = [], []
    for i in range(len(bpe_node_lst)):
        Gi, tagi = [], []
        i_parent = bpe2word[i][0]  # word node
        for Node_index, tag in zip(G2[i_parent], tag2[i_parent]):  # Node_index: output_node
            if Node_index == i_parent:  # Node out-to itself
                Gi.append(i)
                assert tag == ':self', 'Inconsistent tag, should be :self but get {}'.format(tag)
                tagi.append(tag)
            else:  # Node to other node
                if bpe2word[i][1]:  # last children of Node_index
                    for bpe_node in word2bpe[Node_index]:  # Node_index's all children
                        Gi.append(bpe_node)
                        tagi.append(tag)  # same tag
                else:  # bpe_node_i is not the last child of i_parent
                    pass

        G2_new.append(Gi)
        tag2_new.append(tagi)

    return (G1_new, tag1_new, G2_new, tag2_new)


def create_structural_path(G1, G2, tags):
    tag_dict = {}
    for i in range(len(G2)):
        for j, tag in zip(G2[i], tags[i]):
            if i == j:
                tag_dict[(i, j)] = "None"
            else:
                if tag.startswith(':arg'):
                    tag = tag.replace('arg', 'ARG', 1)
                tag_dict[(i, j)] = '-' + tag
                tag_dict[(j, i)] = '+' + tag
    maps = {i: list(set(l[0] + l[1]) - set([i])) for i, l in enumerate(zip(G1, G2))}
    all_paths = {}
    graph = Graph(maps)

    for i in range(len(maps)):
        ith_path = []
        for j in range(len(maps)):
            path = graph.find_path(i, j)
            # pathstr = ''.join([tag_dict[path[a],path[a+1]] for a in range(len(path)-1)]) if len(path)>1 else 'None'
            if not path:
                print(i, j)
            pathstr = [tag_dict[path[a], path[a + 1]] for a in range(len(path) - 1)] if len(path) > 1 else ['None']
            all_paths[(i, j)] = pathstr
            if i == 0:
                EOS_str = ['+:EOS'] + pathstr if pathstr != ['None'] else ['+:EOS']
                all_paths[(len(maps), j)] = EOS_str
        endstr = all_paths[(i, 0)] + ['-:EOS'] if all_paths[(i, 0)] != ['None'] else ['-:EOS']
        all_paths[(i, len(maps))] = endstr
    all_paths[(len(maps), len(maps))] = ['None']
    return all_paths

