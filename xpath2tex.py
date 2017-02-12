#!/usr/bin/env python3

import re
import sys
import os
import argparse
import logging
import itertools
import collections.abc
from collections import defaultdict

from lxml import etree

# Typing
import io
from typing import Sequence, Mapping, Iterator, Generic, AbstractSet, Callable
from typing import Any, List, Tuple
from typing import TypeVar, Union
from typing.io import TextIO
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

EMPTY_ELEMENT = etree.Element('empty')

SORTING_TRANSFORMS = {
    'ident': lambda x: x,
    'ip': lambda x: [int(i) for i in x.split('.')],
    'port': lambda x: int(x.split('/')[0]) if x else -1
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


def query_row(
        row: Element,
        col_xpaths: Sequence[Sequence[XPath]],
        col_relatives: Mapping[XPath, List[int]]) -> List[str]:
    col_relatives = {
        next(iter(xpath(row)), EMPTY_ELEMENT): cols
        for xpath, cols in col_relatives.items()
    }
    rel = {
        col: element
        for element, cols in col_relatives.items()
        for col in cols
    }
    return [
        next(filter(
            None,
            (ensure_str(xpath(rel.get(n, row))) for xpath in xpaths)),
            '')
        for n, xpaths in enumerate(col_xpaths)
    ]


def query_row_group(
        row: Element,
        group: int,
        col_xpaths: Sequence[Sequence[XPath]],
        col_relatives: Mapping[XPath, List[int]]) -> Tuple[List[str], List[List[str]]]:
    grouped, non_grouped = col_xpaths[:group], col_xpaths[group:]
    non_grouped_relative = {
        tuple(xpath(row)): cols
        for xpath, cols in col_relatives.items()
    }
    rel = {
        col: element
        for element, cols in non_grouped_relative.items()
        for col in cols
    }
    return (
        query_row(row, grouped, col_relatives),
        [
            [ensure_str(result) for result in xpath(row)]
            if n not in rel else
            [ensure_str(xpath(rel_row)) for rel_row in rel[n]]
            for n, xpaths in enumerate(non_grouped, group)
            for xpath in xpaths
        ]
    )


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
        row_style: str='%s',
        col_defaults: Mapping[int, str]=None,
        col_formats: Mapping[int, str]=None,
        col_escapes: Mapping[int, bool]=None,
        col_group: int=None,
        skip_cols: AbstractSet[int]=set(),) -> str:
    col_defaults = col_defaults or {}
    col_formats = col_formats or {}
    col_escapes = col_escapes or {}
    return ' & '.join(
        format_item(
            col or col_defaults.get(n, ''),
            col_formats.get(n),
            col_escapes.get(n, True))
        for n, col in enumerate(cols)
        if n not in skip_cols
    ) + r' \\'


def format_row_group(
        group_cols: List[str],
        non_group_cols: List[List[str]],
        **kwargs) -> Iterator[str]:
    assert len(group_cols) > 0
    assert len(non_group_cols) > 0
    n_rows = max(map(len, non_group_cols))
    # print((group_cols, non_group_cols), file=sys.stderr)
    yield r'\multirow[t]{%d}{*}{%s} & %s' % (
        n_rows, group_cols[0], format_row(
            group_cols[1:] +
            [col_rows[0] if col_rows else '' for col_rows in non_group_cols],
            **kwargs))
    for extra_row in itertools.islice(itertools.zip_longest(
            *non_group_cols, fillvalue=''), 1, None):
        yield format_row([''] * len(group_cols) + list(extra_row), **kwargs)
    yield '\\midrule\n'


def enum_rows(
        in_file: TextIO,
        row_xpath: str,
        col_xpaths: Sequence[Union[str, Sequence[str]]],
        row_style: str='%s',
        col_group: int=0,
        sort_by: Tuple[int, Callable[[str], Any]]=None,
        skip_cols: AbstractSet[int]=set(),
        col_relatives: Mapping[str, List[int]]=None,
        col_defaults: Mapping[int, str]=None,
        col_formats: Mapping[int, str]=None,
        col_escapes: Mapping[int, bool]=None) -> Iterator[str]:
    col_defaults = col_defaults or {}
    col_formats = col_formats or {}
    col_escapes = col_escapes or {}
    col_relatives = col_relatives or {}
    row_xpath = XPath(row_xpath)
    col_xpaths = [
            [XPath(i)] if isinstance(i, str) else [XPath(j) for j in i]
            for i in col_xpaths]
    col_relatives = {XPath(k): v for k, v in col_relatives.items()}
    tree = etree.parse(in_file)
    if col_group > 0:
        for row_element in row_xpath(tree):
            group_cols, non_group_cols = query_row_group(
                    row_element, col_group, col_xpaths, col_relatives)
            yield from format_row_group(
                group_cols=group_cols,
                non_group_cols=non_group_cols,
                row_style=row_style,
                col_formats=col_formats,
                col_escapes=col_escapes,
                col_defaults=col_defaults,
                skip_cols=skip_cols,
            )
        return
    row_cols = [
        query_row(row_element, col_xpaths, col_relatives)
        for row_element in row_xpath(tree)]
    if sort_by is not None:
        col_sort, translate = sort_by
        if translate is None:
            translate = SORTING_TRANSFORMS['ident']
        reverse = col_sort < 0
        sort_by = abs(col_sort)
        row_cols.sort(
                key=lambda cols: translate(ensure_str(cols[col_sort])),
                reverse=reverse)
    for cols in row_cols:
        yield format_row(
            cols=cols,
            row_style=row_style,
            col_formats=col_formats,
            col_escapes=col_escapes,
            col_defaults=col_defaults,
            skip_cols=skip_cols
        )


def output_xml(
        in_filename: str,
        col_xpaths: Sequence[Union[str, Sequence[str]]],
        out_file: io.TextIOBase=sys.stdout,
        col_names: Sequence[str]=None,
        row_aligns: Sequence[str]=None,
        print_environment: bool=False,
        skip_cols: AbstractSet[int]=set(),
        environment_name: str='tabular',
        **kwargs) -> None:
    if col_xpaths is None:
        col_xpaths = ['text()']
    if print_environment:
        if row_aligns is None:
            row_aligns = 'l' * (len(col_xpaths) - len(skip_cols))
        print(r'\begin{%s}{%s}\toprule' % (environment_name, row_aligns), file=out_file)
    if col_names is not None:
        print(
            ' & '.join(
                r'\textbf{%s}' % escape_tex(name)
                for n, name in enumerate(col_names)
                if n not in skip_cols
            ) + r'\\\midrule',
            file=out_file)
    with open(in_filename) as f:
        rows = enum_rows(
            f,
            col_xpaths=col_xpaths,
            skip_cols=skip_cols,
            **kwargs)
        for row in rows:
            print(row, file=out_file)
    if print_environment:
        print('\\bottomrule\n\\end{%s}' % environment_name, file=out_file)


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
        cols = [int(i) for i in m.group().split(',')]
        for col in cols:
            if col in col_defaults:
                raise ValueError('Column %d default set more than once' % col)
            col_defaults[col] = text
    return col_defaults


def parse_relatives(relatives: List[str]) -> Mapping[str, List[int]]:
    if relatives is None:
        return {}
    col_relative = defaultdict(list)
    expr = re.compile(r'^(\d+(?:,\d+)*):(.+)$')
    for text in relatives:
        m = expr.match(text)
        if m is None:
            raise ValueError('Column expression %s invalid' % cols)
        cols, xpath = m.groups()
        cols = [int(i) for i in cols.split(',')]
        col_relative[xpath].extend(cols)
    return dict(col_relative)


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
        elif isinstance(val, set):
            current[key] = cval | val
        elif isinstance(val, (str, int)):
            logger.warning('Overwriting key %s' % key)
            current[key] = val
        else:
            current[key] = val
    return current


def get_config(args):
    col_formats, col_escapes = parse_formats(args.formats)
    col_defaults = parse_defaults(args.defaults)
    col_relatives = parse_relatives(args.relatives)
    kwargs = {
        'in_filename': args.file,
        'row_xpath': args.rows,
        'col_xpaths': args.cols,
        'col_formats': col_formats,
        'col_escapes': col_escapes,
        'col_defaults': col_defaults,
        'skip_cols': args.skip_cols,
        'row_style': args.row_style,
        'col_names': args.names,
        'col_relatives': col_relatives,
        'sort_by': args.sort_by,
        'row_aligns': args.align,
        'print_environment': args.print_environment,
        'col_group': args.group,
        'environment_name': args.environment_name,
    }
    if args.rows.startswith('auto:'):
        del kwargs['row_xpath']
        config_file = os.path.join(
                os.path.split(os.path.realpath(__file__))[0],
                'auto', args.rows[5:]) + '.py'
        with open(config_file) as f:
            code_str = f.read()
            config = {}
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
    invalid_col = next((
            col
            for groups in (col_formats, col_escapes, col_defaults)
            for col in groups
            if col is not None and not 0 <= col < n_cols
        ), None)
    if invalid_col is not None:
        raise ValueError('Invalid column number %d' % invalid_col)
    return kwargs


def parse_sort(s):
    parts = s.split(':', 1)
    if len(parts) == 1:
        return int(s), None
    transform = SORTING_TRANSFORMS.get(parts[1])
    if transform is None:
        raise ValueError('Sorting transform %s not known' % parts[1])
    return int(parts[0]), transform


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
            'COLS:=n[,n]')
    parser.add_argument(
            '-g', '--group', default=0, type=int,
            help='Group the first n columns together')
    parser.add_argument(
            '-n', '--names', type=lambda s: s.split(','),
            help='Names of each of the columns, comma separated')
    parser.add_argument(
            '-f', '--format', action='append', dest='formats', default=[],
            help='Set the format for one or all columns.\n'
            'format:=[n[,n]...:]([!]expr|name).\n'
            'n is one or more columns starting from 0, or all if omitted.\n'
            '! specifies to not escape the column.\n'
            'expr is a format expression where %%s is the data' +
            'name is one of: %s.\n' % ', '.join(KNOWN_FORMATS))
    parser.add_argument(
            '-l', '--relative', default=[], action='append', dest='relatives',
            help='The given columns are relative to the given xpath')
    parser.add_argument(
           '-e', '--print-environment', action='store_true',
            help='Print the tabular environment')
    parser.add_argument(
            '--environment-name', default='tabular',
            help='The name of the tabular environment')
    parser.add_argument(
            '-r', '--row-style', default='%s',
            help='The text to insert before each row')
    parser.add_argument(
            '-s', '--skip-cols', default=[],
            type=lambda s: [int(i) for i in s.split(',')],
            help='Skip the specified columns in the output')
    parser.add_argument(
            '--sort-by', type=parse_sort,
            help='Sort by the column, negative to reverse. '
            'By appending :func, the value is transformed first. ' +
            'The current supported transformations are %s' %
            ', '.join(SORTING_TRANSFORMS))
    args = parser.parse_args()
    kwargs = get_config(args)
    output_xml(**kwargs)




if __name__ == '__main__':
    main()

