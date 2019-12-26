import os
from functools import lru_cache

_ip2port = {
    "10.19.90.95": [8501, 8502],
    "10.19.160.33": [8501, 8502],
    "10.19.117.187": [8503, 8504],
    "10.19.128.25": [8503, 8504]
}
_port2ip = {}
for key, value in _ip2port.items():
    for port in value:
        target = _port2ip.get(port, set())
        target.add(key)
        _port2ip[port] = target.copy()

# better to move to config to void leak
_ip2pwd = {
    "10.19.90.95": "Knowbox.cn",
    "10.19.160.33": "Knowbox.cn",
    "10.19.117.187": "root!@#.com",
    "10.19.128.25": "root!@#.com",
}

_address = [
    "ubuntu@10.19.90.95:/data/midas-model",
    "ubuntu@10.19.160.33:/data/midas-model",
    "ubuntu@10.19.117.187:/data/midas-model",
    "ubuntu@10.19.128.25:/data/midas-model",
]

_source = "/data/lishuang/ad_kafka/update-model-1/serving"
_port = [8501, 8502, 8503, 8504]
_state = """sshpass -p {pwd} scp -r {source} {target}"""


def trans_model(version: int, source: str = None, target: str = None, port: int = None) -> None:
    """
    deploy source/version's model to target/port/version ,port decides ips
    :param version: model version
    :param source: model's path
    :param target: deploy the model to where, add port is the abs_path
    :param port: model category
    :return: None
    """

    @lru_cache(maxsize=20)
    def parse_ip(address):
        ip = address.split(":")[0].split("@")[1]
        return ip

    def parse2list(item):
        if not isinstance(item, (list, tuple)):
            item = [item]
        return item

    port = parse2list(port or _port)
    source = source or _source
    addresses = parse2list(target or _address.copy())
    # print(addresses)

    for pp in port:
        ips = _port2ip.get(pp, set())
        if len(ips) == 0:
            continue
        for address in addresses:
            if parse_ip(address) not in _port2ip[pp]:
                continue
            real_source = os.path.join(source, str(version))
            real_target = os.path.join(address, str(pp))

            # TODO: use thread???
            os.system(_state.format(pwd=_ip2pwd[parse_ip(address)],
                                    source=real_source,
                                    target=real_target))


if __name__ == "__main__":
    trans_model(74)
