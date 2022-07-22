import numpy as np

import networkx as ntx
from networkx.algorithms import tree

import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

import random
from itertools import combinations
from math import comb

from matplotlib.widgets import Button
from matplotlib.widgets import Cursor

def get_distance(x1,x2,y1,y2):
    """
    Retorna distância euclidiana dado dois pontos (x1,y1) e (x2,y2)
    """
    dist = ((x1 - x2)**2 + (y1- y2)**2)**(1/2)
    return dist

def generate_random_graph(n, max_degree, seed=10):
    """
    Entrada
    ----------
    n (int):                Número de vértices do grafo
    max_degree (int) [4,n]: Maior grau possível de cada vértice
    seed (int):             Semente do gerador random
    ----------
    Saída:
    G: Grafo                Grafo gerado
    ----------
    """
    #Gera erro caso o grau esperado seja maior que o número de nós do grafo
    if(max_degree > n):
        raise Exception("O grau máximo deve ser menor que o número de vértices")

    #Atribui uma seed para o random
    random.seed(seed)

    #Gera uma lista aleatória com o grau esperado para cada nó, variando no intervalo [4 , max_degree]
    expected_degree = [random.randint(4,max_degree) for _ in range(n)]
    G = ntx.expected_degree_graph(expected_degree, seed=None, selfloops=False)

    #Transforma em um grafo não orientado
    ntx.to_undirected(G)

    #Atribui uma coordenada aleatória para cada vértice, simulando a latitude e longitude pos = (x,y)
    for node in ntx.nodes(G):
        ntx.set_node_attributes(G,{node:{"pos":(random.uniform(0,5),random.uniform(0,5))}})
         
    #Atribui um peso a cada aresta
    for edge in ntx.edges(G):
        ntx.set_edge_attributes(G,{edge:{"weight":round(get_distance(G.nodes[edge[0]]['pos'][0], G.nodes[edge[1]]['pos'][0], G.nodes[edge[0]]['pos'][1], G.nodes[edge[1]]['pos'][1]),2)}})        
    return G

def prim(G):
  """
  Função retorna a árvore geradora mínima utilizando o algoritimo de Prim
  Retorna as arestas que devem ser conectadas para que a distância total seja mínima
  """
  E = tree.minimum_spanning_edges(G, algorithm="prim", data=False)
  edges = list(E)
  sorted_edges = sorted(sorted(x) for x in edges)
  return sorted_edges

def draw_graph(
    G,
    has_edgeslabel = False,
    edgelist=[],
    title = ''
    ):
    """
    Entrada
    --------
    G:                            Grafo a ser desenhado
    has_edgeslabel (Boolean):     Booleano que determina se as arestas dos grafos possuem labels
    edgelist:                     Lista com as arestas que deseja desenhar
    title (str):                  Titulo da imagem gerada
    
    Saída
    --------
    None    
    """
    f,ax = plt.subplots(figsize=(8,6))
    
    #Gera uma lista de coordenadas x e y com a posição de cada vértice
    pos = ntx.get_node_attributes(G,'pos')
    x = []
    y = []
    for i in range(len(pos)):
        x.append(pos[i][0])
        y.append(pos[i][1])

    #Desenha as arestas de acordo com o problema
    if not has_edgeslabel:
        ntx.draw_networkx_edges(G, pos=pos, edge_color='dimgray',alpha=0.35)
    else:
        edge_labels = dict([((tuple(e1), G[e1[0]][e1[1]]["weight"]))
                    for e1 in edgelist])
        ntx.draw_networkx_edges(G,pos=pos, edgelist=edgelist, edge_color='r', label='weight')
        ntx.draw_networkx_edge_labels(G, pos, edge_labels, rotate = True, font_size = 8)
    sc = plt.scatter(x,y,edgecolor={'black'})

    #Adiciona o nome em cima de cada vértice
    eps=0.04
    if not has_edgeslabel:
        for i, txt in enumerate(range(len(pos))):
            plt.annotate(txt, (x[i]+eps, y[i]+eps))

    #Configurações do matplotlib
    plt.title(title)    
    ax.tick_params(left=True, bottom=True, labelleft=True, labelbottom=True)
    ax.grid(True)
    # for child in ax.get_children():
    #     print(child)
    #     print(type(child))

    # annotations = [child for child in ax.get_children() if isinstance(child, matplotlib.text.Text)]
    
    # print(annotations)
    # Set useblit=True on most backends for enhanced performance.
    
    plt.show()

def get_random_list_of_edges(n):
    """
    Gera n combinações de arestas (Excluindo self-loops)
    """
    if comb(n,2) < n*3:
        return list(combinations(range(n), r = 2))
    else:
        return random.sample(list(combinations(range(n), r = 2)),3*n)

def main():
    #Dados iniciais para gerar o grafo aleatório
    n = 20
    n_max_arestas = 8

    #---------------------------------
    G = generate_random_graph(n, n_max_arestas, seed=10)

    #Gera 3n arestas possíveis
    g_random_edges = get_random_list_of_edges(n)

    #Calcula a distância mínima para ir de uma cidade aleatória A para uma cidade aleatória B
    min_distance = []
    for edge in g_random_edges:
        min_distance.append({"edge":edge,"min_dist":ntx.dijkstra_path_length(G,edge[0],edge[1],weight='weight')})
        
    E = prim(G)


    draw_graph(G, has_edgeslabel = True, edgelist=E,title="Grafo com a árvore geradora mínima")
    return

if __name__ == '__main__':
    main()