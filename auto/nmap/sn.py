{
    'row_xpath': '/nmaprun/host',
    'col_xpaths': [
        'address[@addrtype="ipv4"]/@addr',
        'address[@addrtype="mac"]/@addr',
        # SMB OS Discovery part of the smb-os-discovery script
        '(hostnames/hostname/@name |'
        ' hostscript/script[@id="smb-os-discovery"]/elem[@key="fqdn"])[1]',
    ],
    'col_names': [
        'IP Address',
        'MAC Address',
        'Hostname',
    ],
}
