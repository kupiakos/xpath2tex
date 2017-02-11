#!/usr/bin/env python3

import re
import sys
import os
import argparse
import logging
import collections
from lxml import etree

# Typing
import io
from typing import Sequence, Mapping, Iterator, Generic
from typing import Any, List, Tuple, Dict, Generator
from typing import TypeVar
from numbers import Number
from lxml.etree import _Element as Element
from lxml.etree import XPath


TEX_ESCAPES = [
    ('\\', r'\textbackslash'),
    ('&', r'\&'),
    ('%', r'\%'),
    ('$', r'\$'),
    ('#', r'\#'),
    ('_', r'\_'),
    ('{', r'\{'),
    ('}', r'\}'),
    ('~', r'\textasciitilde{}'),
    ('^', r'\textasciicircum{}'),
    (r'\textbackslash', r'\textbackslash{}'),
]

KNOWN_FORMATS = {
    'mono': (r'\texttt{%s}', True),
    'verb': (r'\verb|%s|', False),
    'italic': (r'\textit{%s}', False),
    'math': (r'$ %s $', True),
    'bold': (r'\textbf{%s}', False),
}

logger = logging.getLogger('xpath2tex')

T = TypeVar('T')

class AnyMap(collections.abc.Mapping, Generic[T]):
    def __init__(self, item: T):
        self.item = item

    def __getitem__(self, value: Any) -> T:
        return self.item

    def __len__(self) -> int:
        return 1

    def __iter__(self) -> Iterator[Any]:
        return iter((None,))

    def __repr__(self) -> str:
        return 'AnyMap({:r})'.format(self.item)


def ensure_str(item) -> str:
    """Ensures that the element passed in turns into a string.
    
    First, if given a sequence of length 1, extract the element.
    If given `str`, it retuns `item`.
    If given `bytes`, it decodes it with UTF-8.
    If given `etree._Element`, it gets the internal text.
    If given a number, return `str(item)`. 
    Otherwise, give a warning and return `str(item)`.
    """
    if isinstance(item, Sequence):
        if len(item) == 1:
            item = item[0]
        elif len(item) == 0:
            return ''
        elif not isinstance(item, (str, bytes)):
            return ', '.join(ensure_str(i) for i in item)
    if item is None:
        return ''
    if isinstance(item, str):
        return item
    if isinstance(item, bytes):
        return item.decode()
    if isinstance(item, Element):
        return ensure_str(item.text)
    if isinstance(item, Number):
        return str(item)
    logger.warning('Could not identify string conversion of %r', item)
    return str(item)


def query_row(row: Element, col_xpaths: Sequence[XPath]) -> List[str]:
    return [ensure_str(xpath(row)) for xpath in col_xpaths]


def replace_multiple(s: str, replaces: Sequence[Tuple[str, str]]) -> str:
    for from_, to in replaces:
        s = s.replace(from_, to)
    return s


def escape_tex(s: str) -> str:
    return replace_multiple(s, TEX_ESCAPES)


def format_item(item: str, fmt: str=None, escape: bool=True) -> str:
    if escape:
        item = escape_tex(item)
    if fmt is not None:
        item = fmt % item
    return item


def format_row(
        cols: Sequence[str],
        col_defaults: Mapping[int, str]={},
        col_formats: Mapping[int, str]={},
        col_escapes: Mapping[int, bool]={}) -> str:
    return ' & '.join(
        format_item(
            col or col_defaults.get(n, ''),
            col_formats.get(n),
            col_escapes.get(n, True))
        for n, col in enumerate(cols)
    )


def enum_rows(
        in_file: io.TextIOBase,
        row_xpath: str,
        col_xpaths: Sequence[str],
        row_style: str='%s',
        col_defaults: Mapping[int, str]={},
        col_formats: Mapping[int, str]={},
        col_escapes: Mapping[int, bool]={}) -> Iterator[str]:
    row_xpath = XPath(row_xpath)
    col_xpaths = [XPath(i) for i in col_xpaths]
    tree = etree.parse(in_file)
    for row_element in row_xpath(tree):
        cols = query_row(row_element, col_xpaths)
        yield row_style % format_row(
            cols=cols,
            col_formats=col_formats,
            col_escapes=col_escapes,
            col_defaults=col_defaults,
        )


def output_xml(
        in_filename: str,
        col_xpaths: Sequence[str],
        out_file: io.TextIOBase=sys.stdout,
        col_names: Sequence[str]=None,
        row_aligns: Sequence[str]=None,
        print_environment: bool=False,
        **kwargs) -> None:
    if col_xpaths is None:
        col_xpaths = ['text()']
    if print_environment:
        if row_aligns is None:
            row_aligns = 'l' * len(col_xpaths)
        print(r'\begin{tabular}{%s}\toprule' % row_aligns, file=out_file)
    if col_names is not None:
        print(
            ' & '.join(
                r'\textbf{%s}' % escape_tex(name) for name in col_names
            ) + r'\\\midrule',
            file=out_file)
    with open(in_filename) as f:
        for row in enum_rows(in_filename, col_xpaths=col_xpaths, **kwargs):
            print(row + r' \\', file=out_file)
    if print_environment:
        print('\\bottomrule\n\\end{tabular}', file=out_file)


def parse_formats(formats: List[str]) -> \
        Tuple[Mapping[int, str], Mapping[int, bool]]:
    col_formats = {}
    col_escapes = {}
    expr = re.compile(r'^(?:(\d+(?:,\d+)*):)?(\!?)(.*)$')
    for fmt in formats:
        m = expr.match(fmt)
        if m is None:
            raise ValueError('Format %s is invalid' % fmt)
        cols, no_escape, expr = m.groups()
        if expr in KNOWN_FORMATS:
            expr, escape = KNOWN_FORMATS[expr]
        else:
            escape = not no_escape
        cols = list(map(int, filter(None, (cols or '').split(','))))
        if not cols:
            col_formats = AnyMap(expr)
            col_escapes = AnyMap(escape)
            break
        for col in cols:
            if col in col_formats:
                raise ValueError('Column %d format set more than once' % col)
            col_formats[col] = expr
            col_escapes[col] = escape
    return col_formats, col_escapes


def parse_defaults(defaults: Mapping[str, str]) -> Mapping[int, str]:
    col_defaults = {}
    expr = re.compile(r'^(\d+(?:,\d+)*)$')
    for cols, text in defaults:
        m = expr.match(cols)
        if m is None:
            raise ValueError('Column expression %s invalid' % cols)
        cols = [int(i) for i in m.groups()]
        for col in cols:
            if col in col_defaults:
                raise ValueError('Column %d default set more than once' % col)
            col_defaults[col] = text
    return col_defaults


def merge_config(current, other):
    current = current.copy()
    for key, val in other.items():
        if current.get(key) is None:
            current[key] = val
            continue
        cval = current[key]
        logger.debug('Merging key %s', key)
        if isinstance(val, dict):
            d = cval.copy()
            d.update(val)
            current[key] = d
        elif isinstance(val, list):
            current[key] = cval + val
        elif isinstance(val, (str, int)):
            logger.warning('Overwriting key %s' % key)
            current[key] = val
    return current


def get_config(args):
    col_formats, col_escapes = parse_formats(args.formats)
    col_defaults = parse_defaults(args.defaults)
    kwargs = {
        'in_filename': args.file,
        'row_xpath': args.rows,
        'col_xpaths': args.cols,
        'col_formats': col_formats,
        'col_escapes': col_escapes,
        'col_defaults': col_defaults,
        'row_style': args.row_style,
        'col_names': args.names,
        'row_aligns': args.align,
        'print_environment': args.print_environment,
    }
    if args.rows.startswith('auto:'):
        del kwargs['row_xpath']
        config_file = os.path.join(
                os.path.split(os.path.realpath(__file__))[0],
                'auto', args.rows[5:]) + '.py'
        with open(config_file) as f:
            code_str = f.read()
            try:
                config = eval(compile(code_str, config_file, 'eval'))
            except SyntaxError:
                l = {}
                exec(compile(code_str, config_file, 'exec'), None, l)
                if 'config' not in l:
                    raise NameError('config not defined')
                config = l['config']
        kwargs = merge_config(kwargs, config)
    n_cols = len(kwargs['col_xpaths'] or [])
    if not all(
            0 <= i < n_cols
            for groups in (col_formats, col_escapes, col_defaults)
            for i in groups
            if i is not None):
        raise ValueError('Invalid column number')
    return kwargs


def main():
    parser = argparse.ArgumentParser(
            description='Convert XML and XPath expressions to LaTeX tables')
    parser.add_argument(
            'file',
            help='The file to import')
    parser.add_argument(
            'rows',
            help='The XPath expression to select each row.\n'
            'Alternatively, if it begins with "auto:", it specifies '
            'a configuration file in the autoconfig directory.')
    parser.add_argument(
            'cols', nargs='*',
            help='The XPath expressions relative to rows for each column. '
            'If none are provided, each row has one column. '
            'If the expression for a column is not a string, '
            'it will be converted to one.')
    parser.add_argument(
            '-a', '--align',
            help='Alignment for the columns')
    parser.add_argument(
            '-d', '--default', action='append', dest='defaults',
            nargs=2, metavar=('COLS', 'TEXT'), default=[],
            help='Set the default value for one or all columns.\n'
            'COLS := n[,n]')
    parser.add_argument(
            '-n', '--names', type=lambda s: s.split(','),
            help='Names of each of the columns, comma separated')
    parser.add_argument(
            '-f', '--format', action='append', dest='formats', default=[],
            help='Set the format for one or all columns.\n'
            'format := [n[,n]...:]([!]expr|name)\n'
            'n is one or more columns starting from 0, or all if omitted.\n'
            '! specifies to not escape the column.\n'
            'expr is a format expression where %s is the data' +
            'name is one of: %s.\n')
    parser.add_argument(
           '-e', '--print-environment', action='store_true',
            help='Print the tabular environment')
    parser.add_argument(
            '-r', '--row-style', default='%s',
            help='The text to insert before each row')
    args = parser.parse_args()
    kwargs = get_config(args)
    output_xml(**kwargs)




if __name__ == '__main__':
    main()

