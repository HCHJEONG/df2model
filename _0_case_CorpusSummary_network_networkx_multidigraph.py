# future tasks
# applicable precedents in body
# repealed cases

import re
import os
import pickle
import platform
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from networkx.algorithms import traversal
from matplotlib import font_manager, rc
from pprint import pprint
from tqdm import tqdm

# 아래의 web2df/_3_glaw_casetxt2df.py regex와는 일치하지 않는 문제 
# # (선고)? 부분이 web2df/regex101 경우와 달리 작동하는 문제
# 서울고등법원 1986. 3. 18. 선고 선고85구260 판결 하나를 잡지 못하는 문제 일단 해결
# re_for_case_full_no = r'[ ㄱ-ㅣ가-힣][1-9]?[ ㄱ-ㅣ가-힣]+[ ][0-9]{1,4}\.[ ][0-9]+\.[ ][0-9]+\.[ ]?[ㄱ-ㅣ가-힣]+[ ]>?(선고)?[0-9]{1,4}[ㄱ-ㅣ가-힣]+[0-9\-#ㄱ-ㅣ가-힣, ]+[ #(0-9ㄱ-ㅣ가-힣]+[)]?[ ㄱ-ㅣ가-힣★*]+'
# re_for_case_full_no = r'[ ㄱ-ㅣ가-힣][1-9]?[ ㄱ-ㅣ가-힣]+[ ][0-9]{1,4}\.[ ][0-9]+\.[ ][0-9]+\.[ ]?>?[ㄱ-ㅣ가-힣]+[ ]?>?[ㄱ-ㅣ가-힣]?[ㄱ-ㅣ가-힣]?[0-9]{1,4}[ㄱ-ㅣ가-힣]+[0-9\-#ㄱ-ㅣ가-힣, ]+[ #(0-9ㄱ-ㅣ가-힣]+[)]?[ ㄱ-ㅣ가-힣★*]+'

# 이제 완전히 같게 하였음 !!!!!!!! 
# # (곻보 등이 포함되는 문제
re_for_case_full_no = r'[ ㄱ-ㅣ가-힣][1-9]?[ ㄱ-ㅣ가-힣()]+[ ][0-9]{1,4}\.[ ]?[0-9]+\.[ ]?[0-9]+\.[ ]?[ㄱ-ㅣ가-힣]+[ ]>?(선고|고지)?[0-9]{1,4}[ㄱ-ㅣ가-힣]+[0-9\-#ㄱ-ㅣ가-힣, ]+[ ,()#0-9ㄱ-ㅣ가-힣]+[)]?[ ㄱ-ㅣ가-힣★*]+'

compiled_re_for_case_full_no = re.compile(re_for_case_full_no)

tqdm.pandas()

# 한글 사용시 마이너스 폰트가 깨지는 문제가 발생할수 있으므로 설정변경
plt.rcParams['axes.unicode_minus'] = False
if platform.system() == 'Windows':
  path = "c:/Windows/Fonts/malgun.ttf"
  font_name = font_manager.FontProperties(fname=path).get_name()
  rc('font', family=font_name)
elif platform.system() == 'Darwin':
  rc('font', family='AppleGothic')
elif platform.system() == 'Linux':
  rc('font', family='NanumBarunGothic')
else:
  print('Unknown system...')
plt_options = {
  'node_color': '#1f78b4',
  'node_size': 30,
  'node_shape': 's', # s o ^ > v < d p h 8
  'edge_color': 'r', # ( r, g, b, a)
  'style': 'solid', # solid dashed
  'width': 2,
  'font_size': 12,
  'font_weight': 'bold',
  'font_color': 'k',
  'with_labels': True
}

def get_case_full_no(text):  # from 'case_full_no' field in both dataframes
  global compiled_re_for_case_full_no
  regEx = compiled_re_for_case_full_no
  text = str(text)

  try:
    # ii = regEx.findall(text)[0]
    ii = regEx.search(text).group(0)
  except Exception as e:
    print(text)
    print(e)
    input('....')
  # print('...')
  # print("iii"+text+"iii")
  # input('...')
  ii = ii.replace('*', '').strip()
  ii = ii.replace('★', '').strip()

  try:
    ii = ii.strip()
    if ii[-1] == ')':
        input(f'... inside get case full no ... {ii}')
        if '(공' in ii.strip():
          ii = ii[:ii.rfind('(공')]
        elif '(폐' in ii.strip():
          ii = ii[:ii.rfind('(폐')]
        elif '(변' in ii.strip():
          ii = ii[:ii.rfind('(변')]
        elif '(헌' in ii.strip():
          ii = ii[:ii.rfind('(헌')]
        elif '(같은' in ii.strip():
          ii = ii[:ii.rfind('(같은')]
        elif '(관보' in ii.strip():
          ii = ii[:ii.rfind('(관보')]
        elif '(동지' in ii.strip():
          ii = ii[:ii.rfind('(동지')]
        elif '(항소' in ii.strip():
          ii = ii[:ii.rfind('(항소')]
        elif '(요형' in ii.strip():
          ii = ii[:ii.rfind('(요형')]
        elif '(요민' in ii.strip():
          ii = ii[:ii.rfind('(요민')]
        elif '(대법원판결요지집' in ii.strip():
          ii = ii[:ii.rfind('(대법원판결요지집')]
        elif '(판례카아드' in ii.strip():
          ii = ii[:ii.rfind('(판례카아드')]
        elif '(판결요지집' in ii.strip():
          ii = ii[:ii.rfind('(판결요지집')]
        elif '판결)' in ii.strip():
          ii = ii[:ii.rfind(')')]
        else:
          print('\nget case full no:')
          print('...')
          print(ii.strip())
          ii = ii.strip()[:-1]
          print(ii.strip())
          input('...')
    elif '(공' in ii:
        ii = ii[:ii.rfind('(공')]

    return ii.strip()
  
  except Exception as e:
    print('\nget case full no:')
    print(e)
    print(len(ii))
    print(text)
    print(f'>>>>{ii}<<<<')
    input('...')
    return ''
  # return(text.strip())

def get_case_full_no_list(text):  # from 'precedents' field in both dataframes
  global compiled_re_for_case_full_no
  regEx = compiled_re_for_case_full_no
  text = str(text)
  # print('....')
  # print(text)
  # regEx = re.compile(
  #     r'[ㄱ-ㅣ가-힣]+[ ][0-9]{,4}\.[ ][0-9]+\.[ ][0-9]+\.[ ]?[ㄱ-ㅣ가-힣]+[ ][0-9]{,4}[ㄱ-ㅣ가-힣]+[\s,#\(\)\-0-9ㄱ-ㅣ가-힣]+')  # 사건번호 regex
  resultList = []
  # ies = regEx.findall(text)
  ies = [x.group(0) for x in regEx.finditer(text)]
  # print(ies)
  # print('...')
  for ii in ies:  # (공2008하, 982) 이런 부분을 제거하고 사건번호 부분만 남김
    ii = ii.replace('*', '').strip()
    ii = ii.replace('★', '').strip()

    try:
      ii = ii.strip()
      if ii[-1] == ')':
          if '(공' in ii.strip():
            ii = ii[:ii.rfind('(공')]
          elif '(폐' in ii.strip():
            ii = ii[:ii.rfind('(폐')]
          elif '(변' in ii.strip():
            ii = ii[:ii.rfind('(변')]
          elif '(헌' in ii.strip():
            ii = ii[:ii.rfind('(헌')]
          elif '(같은' in ii.strip():
            ii = ii[:ii.rfind('(같은')]
          elif '(관보' in ii.strip():
            ii = ii[:ii.rfind('(관보')]
          elif '(동지' in ii.strip():
            ii = ii[:ii.rfind('(동지')]
          elif '(항소' in ii.strip():
            ii = ii[:ii.rfind('(항소')]
          
          elif '(집' in ii.strip():
            ii = ii[:ii.rfind('(집')]
          elif '(20' in ii.strip():
            ii = ii[:ii.rfind('(20')]
          elif '(19' in ii.strip():
            ii = ii[:ii.rfind('(19')]
          elif '(요형' in ii.strip():
            ii = ii[:ii.rfind('(요형')]
          elif '(요민' in ii.strip():
            ii = ii[:ii.rfind('(요민')]
          elif '(대법원판결요지집' in ii.strip():
            ii = ii[:ii.rfind('(대법원판결요지집')]
          elif '(판례카아드' in ii.strip():
            ii = ii[:ii.rfind('(판례카아드')]
          elif '(판결요지집' in ii.strip():
            ii = ii[:ii.rfind('(판결요지집')]

          elif '판결)' in ii.strip():
            ii = ii[:ii.rfind(')')]
          else:
            print('\nget case full no list:')
            print('...')
            print(ii.strip())
            # ii = ii.strip()[:-1]
            ii = ii[:ii.rfind('(')]
            print(ii.strip())
            input('...')
      elif '(공' in ii:
        ii = ii[:ii.rfind('(공')]
      resultList.append(ii.strip())
    except Exception as e:
      print('\nget case full no list:')
      print(e)
      print(text)
      print(ies)
      print(f'>>>>{ii}<<<<')
      input('...')
      # 대법원 2010. 4. 29. 선고 2009두22829 판결)
      # 대법원 1975. 4. 8. 선고 74도3323 전원합의체 판결(폐기)
      # 대법원 2000. 9. 5. 선고 99두8800 판결(변경)
      # 대법원 2001. 2. 13. 선고 2000도5725 판결(공보불게재)
      # 대법원 1999. 7. 27. 선고 98다35020 판결(같은 취지)
      # 헌법재판소 1998. 2. 27. 선고 97헌바79 결정(헌공 제26호)
      # resultList.append('')
  return resultList

if __name__ == "__main__":
    
  old_date_str = str(input("가장 최근에 진행했던 판례 업데이트 작업날짜를 적어주세요(예시:20240719 또는 adam):")) # 가장 최근에 수행한 업데이트 작업 날짜   
  date_str = str(input("판례 업데이트 작업을 진행하는 날짜를 적어주세요(예시:20240719):")) # 판례의 업데이트 작업을 시작한 날짜, 시작날짜는 모든 코드에서 통일되어야함.
  
  # Train Or Use the saved
  if 'y' == input("Do you want to train your network model now? (y/n): "):

      # data loading...
      df_glaw_summary_fullest = pd.read_pickle(
          f'..//web2df//saved//{date_str}//df_glaw_corpus//df_glaw_summary_fullest.pickle').reset_index()
      print()
      print(df_glaw_summary_fullest.info())
      # pprint(df_glaw_summary_fullest.head(2).to_dict(orient='records'))
      print()
      summary_fullest_keys = [*df_glaw_summary_fullest]
      print(
          f"There are {len(summary_fullest_keys)} fields in df summary full as follows: \n")
      print(summary_fullest_keys)
      print()
      # df_glaw_summary_full = json.loads(df_glaw_summary_full.to_json(orient = 'records'))

      df_glaw_corpus_fullest = pd.read_pickle(
          f'..//web2df//saved//{date_str}//df_glaw_corpus//df_glaw_corpus_fullest_gmeta_lmeta.pickle').reset_index()
      print()
      print(df_glaw_corpus_fullest.info())
      # pprint(df_glaw_corpus_fullest.head(2).to_dict(orient='records'))
      print()

      # * for unpacking a list and a dictionary key list
      # ** for unpacking a dictionary with key-0value pairs
      # list(df_glaw_corpus_fullest) | df_glaw_corpus_fullest.columns.tolist() | list(df_glaw_corpus_fullest.columns.values)
      corpus_fullest_keys = [*df_glaw_corpus_fullest]

      if 'closing_argument' in corpus_fullest_keys:
          pass
      else:
          df_glaw_corpus_fullest['closing_argument'] = '2072-12-20'

      print(
          f"There are {len(corpus_fullest_keys)} fields in df corpus fullest as follows: \n")
      print(corpus_fullest_keys)
      print()
      # df_glaw_corpus_fullest = json.loads(df_glaw_corpus_fullest.to_json(orient = 'records'))

      input("Press enter to train your network models: ")

      # summaryGraph ################
      # summaryGraph ################
      # summaryGraph ################
      # summaryGraph ################
      # summaryGraph ################
      # summaryGraph ################
      print("1. summaryGraph training as MultiDiGraph...")
      summaryGraph = nx.MultiDiGraph()

      summaryNodeList =\
          list(zip(
              list(df_glaw_summary_fullest['case_full_no'].progress_map(
                  get_case_full_no
                  # , args=(compiled_re_for_case_full_no,)
                  )),
              list(df_glaw_summary_fullest['number'].progress_map(lambda x: str(x))))
          )

      summaryNeighborList =\
          list(df_glaw_summary_fullest['precedents'].progress_map(
              get_case_full_no_list
              # , args=(compiled_re_for_case_full_no,)
              ))
      
      input(f"Please check: {len(summaryNeighborList)}==={len(summaryNodeList)}...? Press enter in case of true...")

      # list(zip(summaryNodeList, summaryNeighborList)) -> list[tuple(tuple(string, number), list[string])]
      for i, n in tqdm(enumerate(list(zip(summaryNodeList, summaryNeighborList)))): 
          if i % 1000 == 1:
              print()
              print("summaryGraph=================") # n : tuple(tuple(string, number), list[string])
              print(str(i) + " " + n[0][0]) # n[0][0] => case full no
              print("from: ")
              print(n[1]) # n[1] => list of case full no
              print("to: ")
              print(n[0][0])

          if len(n[1]) == 0:
              # input(f'something wrong: {n[1]}')
              continue
          tupleList = []
          for j in n[1]:
              tupleList.append((j.strip(),
                                n[0][0],
                                {"fromItemNumbr": n[0][1]}))
              # list of tuple(선례 판례인용표기, 해당 판례인용표기, dict{판시사항번호} )
              # >>> keys = G.add_edges_from([(4, 5, dict(route=282)), (4, 5, dict(route=37))])
              # >>> G[4]
              # AdjacencyView({5: {0: {}, 1: {'route': 282}, 2: {'route': 37}}})

          summaryGraph.add_edges_from(tupleList)

      print("summrayGraph number of edges:")
      print(summaryGraph.number_of_edges())
      print()

      input('press any key to save graphSummary...')
      if not (os.path.isdir(f'.//model//{date_str}//')):
          os.makedirs(os.path.join(f'.//model//{date_str}//'))
      with open(f'.//model//{date_str}//graphSummary.pickle', 'wb') as f:
          pickle.dump(summaryGraph, f, pickle.HIGHEST_PROTOCOL)

      # corpusGraph ################
      # corpusGraph ################
      # corpusGraph ################
      # corpusGraph ################
      # corpusGraph ################
      # corpusGraph ################
      print("2. corpusGraph training as MultiDiGraph...")
      corpusGraph = nx.MultiDiGraph()

      corpusNodeList =\
          list(df_glaw_corpus_fullest['case_full_no'].progress_map(
              get_case_full_no
              # , args=(compiled_re_for_case_full_no,)
              ))

      corpusNeighborList =\
          list(df_glaw_corpus_fullest['applicable_precedents'].progress_map(
              get_case_full_no_list
              # , args=(compiled_re_for_case_full_no,)
              ))
      
      input(f"Please check: {len(corpusNeighborList)}==={len(corpusNodeList)}...? Press enter in case of true...")
      # list(zip(corpusNodeList, corpusNeighborList)) -> list[tuple(string, list[string])]
      for i, n in tqdm(enumerate(list(zip(corpusNodeList, corpusNeighborList)))): 

          if i % 1000 == 1:
              print()
              print("corpusGraph=================")
              print(str(i) + " " + n[0]) # n[0] => case full no
              print("from: ")
              print(n[1]) # n[1] => list of case full no
              print("to: ")
              print(n[0])

          if len(n[1]) == 0:
              # input(f'something wrong: {n[1]}')
              continue
          tupleList = []
          for j in n[1]:
              tupleList.append((j.strip(), n[0]))
              # list of tuple(선례 판례인용표기, 해당 판례인용표기)

          corpusGraph.add_edges_from(tupleList)

      print("corpusGraph number of edges:")
      print(corpusGraph.number_of_edges())
      print()
      
      input('press any key to save graphCorpus...')
      if not (os.path.isdir(f'.//model//{date_str}//')):
          os.makedirs(os.path.join(f'.//model//{date_str}//'))
      with open(f'.//model//{date_str}//graphCorpus.pickle', 'wb') as f:
          pickle.dump(corpusGraph, f, pickle.HIGHEST_PROTOCOL)

  else:

      with open(f'.//model//{date_str}//graphSummary.pickle', 'rb') as f:
          summaryGraph = pickle.load(f)
      with open(f'.//model//{date_str}//graphCorpus.pickle', 'rb') as f:
          corpusGraph = pickle.load(f)
      print()

  q = input('2 models loaded... press enter to continue...(quit for q)')
  if q == 'q':
     quit()
  print()
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################

  # connected components / spanning tree visualizing
  print("\nsummaryGraph: \n")
  print(summaryGraph)
  print()
  print("\nQuantity of its weakly connected compoenets:", nx.number_weakly_connected_components(summaryGraph))
  print(len(list(nx.connected_components(nx.Graph(summaryGraph)))))
  # print(traversal.dfs_edges(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결'))
  print("\'대법원 2010. 7. 15. 선고 2010도2527 판결\' Depth First Search Tree:")
  print(traversal.dfs_tree(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결')) # DiGraph with 43 nodes and 42 edges
  print(type(traversal.dfs_tree(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결'))) # <class 'networkx.classes.digraph.DiGraph'>
  print(traversal.dfs_tree(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결').graph) # {}
  print(nx.to_dict_of_dicts(traversal.dfs_tree(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결')))
  print()
  print("\'대법원 2010. 7. 15. 선고 2010도2527 판결\' Successors:")
  print(summaryGraph.successors('대법원 2010. 7. 15. 선고 2010도2527 판결')) # <dict_keyiterator object at 0x000002A0D054AF40>
  print(list(summaryGraph.successors('대법원 2010. 7. 15. 선고 2010도2527 판결'))) # ['대법원 2010. 10. 14. 선고 2010도387 판결']
  print()
  print("\'대법원 2010. 7. 15. 선고 2010도2527 판결\' Depth First Search Successors:")
  print(traversal.dfs_successors(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결'))
  print(nx.dfs_successors(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결')) # ditto
  print(type(traversal.dfs_successors(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결'))) # <class 'dict'>
  print()
  print("\'대법원 2010. 7. 15. 선고 2010도2527 판결\' Depth First Search Reverse Successors Graph:")
  print(traversal.dfs_successors(summaryGraph.reverse(), '대법원 2010. 7. 15. 선고 2010도2527 판결'))
  print()
  print("\'대법원 2010. 7. 15. 선고 2010도2527 판결\' Depth First Search Predecessors:")
  print(traversal.dfs_predecessors(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결'))
  print(nx.dfs_predecessors(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결'))
  print(type(traversal.dfs_predecessors(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결')))
  print()
  # print(nx.weakly_connected_components(summaryGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결'))
  # print 내 , 는 space를 생성함
  print('\nQuantity of edges with attribute of \'fromItemNumbr\':', len(list(summaryGraph.edges.data('fromItemNumbr'))))
  for u, v, fromItemNumbr in summaryGraph.edges(data="fromItemNumbr"):
      if fromItemNumbr is not None and float(fromItemNumbr) > 17.0:
          print("from", u)
          print("to", v)
          print(fromItemNumbr)
          # from 대법원 2010. 7. 15. 선고 2010도2527 판결
          # to 대법원 2010. 10. 14. 선고 2010도387 판결
  print("from 대법원 2010. 7. 15. 선고 2010도2527 판결",
        summaryGraph.has_node("대법원 2010. 7. 15. 선고 2010도2527 판결"))
  # 대법원 2010. 10. 14. 선고 2010도387 판결"
  print("to", list(summaryGraph.successors("대법원 2010. 7. 15. 선고 2010도2527 판결")))
  print(summaryGraph.has_edge("대법원 2010. 7. 15. 선고 2010도2527 판결",
        "대법원 2010. 10. 14. 선고 2010도387 판결"))
  print()
  print("\nIs there any predecessor of \'대법원 2010. 7. 15. 선고 2010도2527 판결\'?", list(
      summaryGraph.predecessors("대법원 2010. 7. 15. 선고 2010도2527 판결")))
  print(summaryGraph.has_edge("대법원 1998. 8. 21. 선고 96도2340 판결", "대법원 2010. 7. 15. 선고 2010도2527 판결"))
  print()
  print("\nCheck node in summaryGraph one by one:")
  for node in summaryGraph:
      print(node)
      if 'q' == input('quit or not? (q/ENTER): '):
          break
  print("\nCheck adjacent list in summaryGraph one by one:")
  for node, nbrsdict in summaryGraph.adj.items():
      print(node)
      pprint(nbrsdict)
      if 'q' == input('quit or not? (q/ENTER): '):
          break
  print("\nCheck edge in summaryGraph one by one:")
  for e, datadict in summaryGraph.edges.items():
      print(e)
      pprint(datadict)
      if 'q' == input('quit or not? (q/ENTER): '):
          break
  print()

#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################
#####################################################################

  print("\ncorpusGraph: \n")
  print(corpusGraph)
  print()
  tempList4 = []
  tempList3 = []
  tempList2 = []
  tempList1 = []
  tempList0 = []
  for i in corpusGraph.adjacency():
      m, n = i  # n <- successor nodes dictionary
      for k, l in n.items():
          # print(l[0])
          # # l <- successor node의 edges의 index들을 키로 가진 dictionary
          # Multi Graph이므로 edge는 0부터 번호를 가질 수 있음
          if 4 < len(l.keys()):
              tempList4.append(l)
          elif 3 in l.keys():
              tempList3.append(l)
          elif 2 in l.keys():
              tempList2.append(l)
          elif 1 in l.keys():
              tempList1.append(l)
          else:
              tempList0.append(l)
  print("\nQuantity of its weakly connected components:")
  print(nx.number_weakly_connected_components(corpusGraph))
  print(len(list(nx.connected_components(nx.Graph(corpusGraph)))))
  # print(nx.weakly_connected_components(corpusGraph, '대법원 2010. 7. 15. 선고 2010도2527 판결'))
  # print 내 , 는 space를 생성함
  print('Not Multi:', len(tempList0))
  print('Double   :', len(tempList1))
  print('Triple   :', len(tempList2))
  print('Quadraple:', len(tempList3))
  print('More     :', len(tempList4))
  print()
  # from 대법원 2010. 7. 15. 선고 2010도2527 판결
  # to 대법원 2010. 10. 14. 선고 2010도387 판결
  print("from 대법원 2010. 7. 15. 선고 2010도2527 판결",
        corpusGraph.has_node("대법원 2010. 7. 15. 선고 2010도2527 판결"))
  # 대법원 2010. 10. 14. 선고 2010도387 판결"
  print("to", list(corpusGraph.successors("대법원 2010. 7. 15. 선고 2010도2527 판결")))
  print("\nEdge between exists or not:", corpusGraph.has_edge("대법원 2010. 7. 15. 선고 2010도2527 판결",
        "대법원 2010. 10. 14. 선고 2010도387 판결"))
  print()
  print("Is there any predecessor of \'대법원 2010. 7. 15. 선고 2010도2527 판결\'?", list(
      corpusGraph.predecessors("대법원 2010. 7. 15. 선고 2010도2527 판결")))
  # print(corpusGraph.has_edge("대법원 2010. 10. 14. 선고 2010도387 판결", "대법원 2010. 7. 15. 선고 2010도2527 판결"))
  print()
  print("\ncheck node in corpusGraph one by one:")
  for node in corpusGraph:
      print(node)
      if 'q' == input('quit or not? (q/ENTER): '):
          break
  print("\ncheck adjacent list in corpusGrpah one by one:")
  for node, nbrsdict in corpusGraph.adj.items():
      print(node)
      pprint(nbrsdict)
      if 'q' == input('quit or not? (q/ENTER): '):
          break
  print("\ncheck edge in corpusGraph one by one:")
  for e, datadict in corpusGraph.edges.items():
      print(e)
      pprint(datadict)
      if 'q' == input('quit or not? (q/ENTER): '):
          break
  
  # Visualization
  # G = nx.complete_graph(5)
  tempList = ["대법원 2010. 7. 15. 선고 2010도2527 판결"]
  for node in summaryGraph.predecessors("대법원 2010. 7. 15. 선고 2010도2527 판결"):
      tempList.append(node)
  print(tempList)
  
  G = summaryGraph.subgraph(tempList)
  layouts = {'spring': nx.spring_layout(G), 
          'spectral':nx.spectral_layout(G), 
          'shell':nx.shell_layout(G), 
          'fruchterman_reingold':nx.layout.fruchterman_reingold_layout(G), 
          'kamada_kawai':nx.kamada_kawai_layout(G), 
          'random':nx.random_layout(G)
        } 
  pos = layouts['kamada_kawai']    
  nx.draw_networkx(G, pos = pos, **plt_options)
  # plt.savefig('network.png')
  plt.show()
