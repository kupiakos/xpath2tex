{
    'row_xpath': '/scanning_report/hosts/host',
    'col_xpaths': [
        'result[@name="%s" and text() != "[n/a]"]' % name
        for name in (
            'IP',
            'MAC Address',
            'Hostname',
            'Ping',
        )
    ],
    'col_names': [
        'IP Address',
        'MAC Address',
        'Hostname',
        'Ping',
    ],
    'sort_by': (0, SORTING_TRANSFORMS['ip']),
}
