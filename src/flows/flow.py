"""
flows/flow.py
NetworkFlow data class — represents a single bidirectional network flow.

A flow is defined by its 5-tuple key:
    (src_ip, dst_ip, src_port, dst_port, protocol)

Packets are accumulated until one of:
  - A FIN/RST TCP flag closes the flow
  - The flow timeout expires (default: 120s idle, 600s absolute)
  - The flow manager flushes it for export
"""

from dataclasses import dataclass, field
from typing import List
import time


@dataclass
class PacketRecord:
    timestamp:   float   # epoch seconds (time.time())
    length:      int     # total IP payload length in bytes
    direction:   str     # "fwd" (src->dst) or "bwd" (dst->src)
    tcp_flags:   int     # raw TCP flags byte (0 if UDP/ICMP)
    header_len:  int     # IP + transport header length in bytes
    push_flag:   bool    # TCP PSH flag set
    urg_flag:    bool    # TCP URG flag set
    window_size: int     # TCP window size (0 if not TCP)


@dataclass(frozen=True)
class FlowKey:
    src_ip:   str
    dst_ip:   str
    src_port: int
    dst_port: int
    protocol: int   # 6=TCP, 17=UDP, 1=ICMP


@dataclass
class NetworkFlow:
    key:        FlowKey
    start_time: float = field(default_factory=time.time)
    last_seen:  float = field(default_factory=time.time)

    # Packet lists (forward = src->dst, backward = dst->src)
    fwd_packets: List[PacketRecord] = field(default_factory=list)
    bwd_packets: List[PacketRecord] = field(default_factory=list)

    # TCP handshake window sizes (first packet only)
    init_win_fwd: int = 0
    init_win_bwd: int = 0

    # Flow lifecycle flags
    is_finished: bool = False   # set on FIN/RST

    def add_packet(self, pkt: PacketRecord) -> None:
        """Add a packet to the appropriate direction list and update timestamps."""
        if not self.fwd_packets and not self.bwd_packets:
            self.start_time = pkt.timestamp
        self.last_seen = pkt.timestamp
        if pkt.direction == "fwd":
            if not self.fwd_packets:
                self.init_win_fwd = pkt.window_size
            self.fwd_packets.append(pkt)
        else:
            if not self.bwd_packets:
                self.init_win_bwd = pkt.window_size
            self.bwd_packets.append(pkt)

    @property
    def duration_us(self) -> float:
        """Flow duration in microseconds (matches CICIDS2017 Flow Duration feature)."""
        return max((self.last_seen - self.start_time) * 1_000_000, 0.0)

    @property
    def total_packets(self) -> int:
        return len(self.fwd_packets) + len(self.bwd_packets)

    @property
    def total_bytes(self) -> int:
        return (
            sum(p.length for p in self.fwd_packets)
            + sum(p.length for p in self.bwd_packets)
        )

    def mark_finished(self) -> None:
        self.is_finished = True

    def is_idle(self, idle_timeout: float = 120.0) -> bool:
        return (time.time() - self.last_seen) > idle_timeout

    def is_expired(self, absolute_timeout: float = 600.0) -> bool:
        return (time.time() - self.start_time) > absolute_timeout
