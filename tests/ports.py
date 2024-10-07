import socket

import pytest


def check_port(port: int) -> bool:
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with tcp, udp:
        try:
            tcp.bind(("127.0.0.1", port))
            udp.bind(("127.0.0.1", port))
        except OSError:
            return False
        else:
            return True


def check_port_range(port_range: range) -> bool:
    return all(check_port(port) for port in port_range)


class Ports:
    NEXT_PORT = 10000

    def allocate(self, num: int) -> int:
        """
        Allocates
        """
        while Ports.NEXT_PORT + num <= 65535:
            start = Ports.NEXT_PORT
            Ports.NEXT_PORT += num
            if not check_port_range(range(start, Ports.NEXT_PORT)):
                continue
            return start
        msg = "cannot find enough free port"
        raise OSError(msg)


@pytest.fixture
def ports() -> Ports:
    return Ports()
