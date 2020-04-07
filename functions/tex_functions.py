import subprocess
from pathlib import Path
import pandas as pd
from functions import file_functions
from pylatexenc import latexencode

"""
Library for the writing as and production of LaTeX typeset.
"""


# noinspection SpellCheckingInspection
class MultipleTablesDocument:

    def __init__(self, destination, centering=True):
        self.text = '\n'.join([r'\documentclass{article}', r'\usepackage[center]{titlesec}', r'\usepackage{booktabs}',
                               r'\usepackage[section]{placeins}', r'\usepackage{caption}',
                               r'\usepackage[margin=0.5in]{geometry}', r'\begin{document}', ])
        self.destination = Path(destination)
        self.centering = centering

    def add(self, df, caption=None, header=True, index=True, **kwargs):
        # OPTIMIZE : see if we can reduce whitespace from these tables

        assert isinstance(df, pd.DataFrame)

        self.text += df_to_table(df, caption=caption, label=None, header=header,
                                 index=index, centering=self.centering, remove_table_number=True, **kwargs)

        return self

    def produce(self):
        self.text += '\n\n' + r'\end{document}'
        Path(self.destination).parent.mkdir(parents=True, exist_ok=True)
        latex_str_2_pdf(self.text, self.destination)


def df_2_pdf(df, pdf, **kwargs):
    latex_str_2_pdf(df_2_standalone_latex_str(df, **kwargs), pdf)


def latex_str_2_pdf(latex_str, target_file, keep_tex=False):
    target_file = Path(target_file)
    tex_file = target_file.parent / (target_file.name.replace('.pdf', '.tex'))
    with open(tex_file, 'w+') as wf:
        wf.write(latex_str)
    tex_2_pdf(tex_file, target_file)
    if not keep_tex:
        file_functions.delete(tex_file)


def tex_2_pdf(tex_file, pdf_file):
    # Apparently, you can't just write anywhere. This is now solved by creating a temporary pdf file, and then
    # copying it to the desired location.
    out_ext = 'pdf'
    temp_pdf = Path('temp.' + out_ext)
    ext_less = Path('temp')

    process = subprocess.Popen([
        'latex',
        '-output-format=' + out_ext,
        '-job-name=' + str(ext_less),
        str(tex_file)])
    process.wait()

    # cleanup
    for ext in ['aux', 'log']:
        file_functions.delete(ext_less.parent / (ext_less.name + '.' + ext))
    file_functions.copyfile(temp_pdf, pdf_file)
    file_functions.delete(temp_pdf)


# noinspection SpellCheckingInspection
def df_2_standalone_latex_str(df, **kwargs):
    # TODO combine with df_to_table
    pre = r'''\documentclass[convert]{standalone}
    \usepackage{booktabs}
    \begin{document}'''

    post = r'\end{document}'

    if isinstance(df, pd.DataFrame):
        df = df.to_latex(**kwargs)
    return pre + '\n' + df + '\n' + post


def df_to_table(df, caption=None, label=None, centering=True, floating='h', fn_out=None,
                remove_table_number=False, max_string_length=1000, multicolumn=False,
                add_phantom=False,
                phantom_length=4,
                phantom_column_position=0,
                **kwargs):
    # TODO pandas 1.0.0 also adds label and caption
    assert isinstance(df, pd.DataFrame)

    if add_phantom:
        phantom_column = f'\\phantom{{{"x" * phantom_length}}}'
        if isinstance(df.columns, pd.MultiIndex):
            phantom_column = (phantom_column,) * df.columns.nlevels
        columns = list(df.columns)[:phantom_column_position] + [phantom_column] + list(df.columns)[
                                                                                  phantom_column_position:]
        df[phantom_column] = ''
        df = df[columns]

    s = r'\begin{table'
    if multicolumn:
        s += '*'
    s += '}[' + floating + ']\n'

    if centering:
        s += r'\centering' + '\n'

    kwargs.setdefault('escape', True)
    with pd.option_context("max_colwidth", max_string_length):
        s += df.to_latex(**kwargs)

    if caption:
        s += r'\caption'
        s += '' if not remove_table_number else '*'
        s += '{'
        if kwargs['escape']:
            s += latexencode.unicode_to_latex(caption)
        else:
            s += caption
        s += '}\n'

    if label:
        s += r'\label{' + label + '}\n'

    s += r'\end{table'
    if multicolumn:
        s += '*'
    s += r'}'

    if fn_out:
        with open(fn_out, 'w+') as wf:
            wf.write(s)
    else:
        return s
