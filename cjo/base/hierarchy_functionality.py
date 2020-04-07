# The long-term plan is to make the hierarchy its own class; these functions will be in there
import pandas as pd
from pyyed import Graph
import data.data_loader
from functions import tex_functions


def to_latex_table(hierarchy, fn_out, **kwargs):
    if isinstance(hierarchy, str):
        hierarchy = data.data_loader.generic_hierarchy(hierarchy)

    assert isinstance(hierarchy, dict)
    df = pd.DataFrame(columns=['Categories'])
    import string
    for k, v in hierarchy.items():
        df.loc[k, 'Categories'] = ", ".join([string.capwords(vi).replace('Np', 'NP') for vi in v])
    df.index.name = 'Super-Category'
    df.reset_index(drop=False, inplace=True)

    kwargs.setdefault('column_format', 'lp{3.5in}')
    kwargs.setdefault('caption', 'Hierarchy used in the experiments. (N)P stands for (Non-)Perishable')
    kwargs.setdefault('index', False)
    string = tex_functions.df_to_table(df, **kwargs)
    with open(fn_out, 'w+') as wf:
        wf.write(string)


def hierarchy_to_graphml(h, fn_out):
    node_width = 300
    node_height = 60
    sc_sc_x_sep = 30
    sc_h_y_sep = 60
    c_cs_x_sep = 30
    c_c_y_sep = 15
    edge_x_shift = -c_cs_x_sep // 2

    g = Graph()

    def add_node(name, x, y):
        g.add_node(name, x=str(x), y=str(y), width=str(node_width), height=str(node_height), shape_fill='#FFFFFF',
                   font_size='20')

    x_hierarchy = (len(h) * node_width + (len(h) - 1) * sc_sc_x_sep) // 2 - node_width // 2
    add_node('Hierarchy', x_hierarchy, 0)
    for i, k in enumerate(sorted(h.keys())):
        v = h[k]
        sc_x = (node_width + sc_sc_x_sep) * i
        sc_y = (node_height + sc_h_y_sep)
        add_node(k, sc_x, sc_y)
        g.add_edge('Hierarchy', k, path=[(x_hierarchy + node_width // 2, node_height + sc_h_y_sep // 2),
                                         (sc_x + node_width // 2, sc_y - sc_h_y_sep // 2)])
        for j, vj in enumerate(sorted(v)):
            c_x = sc_x + c_cs_x_sep
            c_y = sc_y + (node_height + c_c_y_sep) * (j + 1)
            add_node(vj, x=c_x, y=c_y)
            g.add_edge(k, vj, path=[(sc_x - edge_x_shift, sc_y + node_height // 2),
                                    (sc_x - edge_x_shift, c_y + node_height // 2)])

    with open(fn_out, 'w+') as wf:
        wf.write(g.get_graph())
