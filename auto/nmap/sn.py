{
    'row_xpath': '/nmaprun/host',
    'col_xpaths': [
        'address[@addrtype="ipv4"]/@addr',
        'address[@addrtype="mac"]/@addr',
        # SMB OS Discovery part of the smb-os-discovery script
        # nbstat covered as well
        [
            'hostnames/hostname/@name',
            'hostscript/script[@id="smb-os-discovery"]/elem[@key="fqdn"]',
            '(hostscript/script[@id="nbstat"]/table[@key="names"]'
                '//elem[@key="name"][1])[1]',
        ],
    ],
    'col_names': [
        'IP Address',
        'MAC Address',
        'Hostname',
    ],
    'sort_by': (0, SORTING_TRANSFORMS['ip']),
}
