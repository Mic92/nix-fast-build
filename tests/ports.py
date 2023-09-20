#!/usr/bin/env python3

import socket

import pytest

NEXT_PORT = 10000


def check_port(port: int) -> bool:
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    with tcp, udp:
        try:
            tcp.bind(("127.0.0.1", port))
            udp.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def check_port_range(port_range: range) -> bool:
    for port in port_range:
        if not check_port(port):
            return False
    return True


class Ports:
    def allocate(self, num: int) -> int:
        """
        Allocates
        """
        global NEXT_PORT
        while NEXT_PORT + num <= 65535:
            start = NEXT_PORT
            NEXT_PORT += num
            if not check_port_range(range(start, NEXT_PORT)):
                continue
            return start
        raise Exception("cannot find enough free port")


@pytest.fixture
def ports() -> Ports:
    return Ports()
