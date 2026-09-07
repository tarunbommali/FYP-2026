"""
capture/packet_capture.py
Scapy-based packet sniffer — the entry point for live traffic ingestion.

Responsibilities
----------------
  - Capture raw packets from a network interface via Npcap (Windows) / libpcap (Linux)
  - Parse each packet into a PacketRecord + FlowKey
  - Hand off to FlowManager.process_packet()
  - Run a background thread to flush timed-out flows every FLUSH_INTERVAL seconds

Dependencies
------------
    pip install scapy

On Windows, Npcap must be installed: https://npcap.com/

Usage
-----
    from capture.packet_capture import PacketCapture

    def on_flow_ready(flow):
        features = extract_features(flow)
        result   = predict_flow(features)
        print(result)

    cap = PacketCapture(interface="Ethernet", on_flow_complete=on_flow_ready)
    cap.start()            # blocks until cap.stop() is called
"""

import logging
import threading
import time
from typing import Callable, Optional

from flows.flow import FlowKey, PacketRecord
from flows.flow_manager import FlowManager
from monitoring import metrics_registry as reg

logger = logging.getLogger(__name__)

# How often to check for timed-out flows (seconds)
FLUSH_INTERVAL: float = 10.0

# TCP/IP protocol numbers
_PROTO_TCP  = 6
_PROTO_UDP  = 17
_PROTO_ICMP = 1


class PacketCapture:
    """
    Live packet capture using Scapy + Npcap.

    Parameters
    ----------
    interface        : str
        Network interface name (e.g. "Ethernet", "Wi-Fi", "eth0").
        Use scapy.arch.get_if_list() to enumerate available interfaces.
    on_flow_complete : Callable[[NetworkFlow], None]
        Callback invoked when a flow is complete or timed out.
    filter_bpf       : str, optional
        BPF filter string (e.g. "tcp or udp"). Default captures all IP traffic.
    idle_timeout     : float
        Seconds of inactivity before a flow is exported.
    absolute_timeout : float
        Maximum lifetime of any flow in seconds.
    """

    def __init__(
        self,
        interface:        str,
        on_flow_complete: Callable,
        filter_bpf:       str   = "ip",
        idle_timeout:     float = 120.0,
        absolute_timeout: float = 600.0,
    ) -> None:
        self._filter    = filter_bpf
        self._running   = False
        self._flush_thread: Optional[threading.Thread] = None

        self._interface = self._resolve_interface(interface)

        self._flow_manager = FlowManager(
            on_flow_complete = on_flow_complete,
            idle_timeout     = idle_timeout,
            absolute_timeout = absolute_timeout,
        )

    def _resolve_interface(self, iface: str) -> str:
        """
        Validate and resolve interface names.
        If specified interface is invalid, automatically fallback to default active interface.
        """
        try:
            # pyrefly: ignore [missing-import]
            from scapy.arch import get_if_list
            # pyrefly: ignore [missing-import]
            from scapy.all import conf
        except ImportError:
            raise ImportError(
                "Scapy is not installed. Install with:\n"
                "    pip install scapy\n"
                "Npcap also required on Windows: https://npcap.com/"
            )

        def _is_present(name: str) -> bool:
            try:
                if name in get_if_list():
                    return True
                for item in conf.ifaces.values():
                    if name == item.name or name == item.description:
                        return True
            except Exception:
                pass
            return False

        # If it already exists, use it directly
        if _is_present(iface):
            return iface

        # Fallback 1: Try default active routing gateway interface
        try:
            default_iface = conf.route.route('0.0.0.0')[0]
            if default_iface and _is_present(default_iface):
                logger.warning(
                    "Network interface '%s' not found. Falling back to default active interface: '%s'",
                    iface, default_iface
                )
                return default_iface
        except Exception:
            pass

        # Fallback 2: Try loopback interfaces
        try:
            for name in ["NPF_Loopback", "Software Loopback Interface 1"]:
                if _is_present(name):
                    logger.warning(
                        "Network interface '%s' not found. Falling back to loopback interface: '%s'",
                        iface, name
                    )
                    return name
        except Exception:
            pass

        raise ValueError(
            f"Network interface '{iface}' is invalid or not found. "
            f"Please run 'venv\\Scripts\\python.exe src\\main.py --list-interfaces' to list valid adapters."
        )

    # -----------------------------------------------------------------------
    def start(self, packet_count: int = 0) -> None:
        """
        Start packet capture (blocking).

        Parameters
        ----------
        packet_count : int
            Number of packets to capture (0 = unlimited).
        """
        # Import Scapy here to allow the module to load even if Scapy is absent
        try:
            # pyrefly: ignore [missing-import]
            from scapy.all import sniff
        except ImportError:
            raise ImportError(
                "Scapy is not installed. Install with:\n"
                "    pip install scapy\n"
                "Npcap also required on Windows: https://npcap.com/"
            )

        self._running = True
        reg.capture_running.set(1)
        self._start_flush_thread()

        logger.info(
            "Capture started | interface=%s | filter='%s'",
            self._interface, self._filter,
        )

        try:
            sniff(
                iface   = self._interface,
                filter  = self._filter,
                prn     = self._handle_packet,
                count   = packet_count,
                store   = False,        # do not buffer packets in RAM
                stop_filter = lambda _: not self._running,
            )
        finally:
            self._running = False
            reg.capture_running.set(0)
            self._flow_manager.flush_all()
            logger.info("Capture stopped.")

    def stop(self) -> None:
        """Signal the capture loop to stop after the next packet."""
        self._running = False

    @property
    def active_flows(self) -> int:
        return self._flow_manager.active_flow_count

    # -----------------------------------------------------------------------
    def _handle_packet(self, raw_pkt) -> None:
        """Scapy callback — called for every captured packet."""
        try:
            pkt_record, flow_key = _parse_packet(raw_pkt)
            if pkt_record is not None and flow_key is not None:
                logger.info(
                    "PacketParsed: %s:%d -> %s:%d | proto=%d | len=%d",
                    flow_key.src_ip, flow_key.src_port,
                    flow_key.dst_ip, flow_key.dst_port,
                    flow_key.protocol, pkt_record.length
                )
                reg.packets_processed_total.inc()
                reg.packets_total.inc()
                self._flow_manager.process_packet(pkt_record, flow_key)
        except Exception as exc:
            logger.warning("Packet parse error: %s", exc)

    def _start_flush_thread(self) -> None:
        def _flush_loop():
            while self._running:
                time.sleep(FLUSH_INTERVAL)
                if self._running:
                    self._flow_manager.flush_expired()

        self._flush_thread = threading.Thread(
            target=_flush_loop, daemon=True, name="flow-flush"
        )
        self._flush_thread.start()


# ---------------------------------------------------------------------------
# Packet parser — converts a Scapy packet into (PacketRecord, FlowKey)
# ---------------------------------------------------------------------------

def _parse_packet(raw_pkt) -> "tuple[Optional[PacketRecord], Optional[FlowKey]]":
    """
    Parse a raw Scapy packet into a PacketRecord and FlowKey.
    Returns (None, None) for non-IP or unsupported packets.
    """
    try:
        # pyrefly: ignore [missing-import]
        from scapy.layers.inet import IP, TCP, UDP, ICMP
    except ImportError:
        return None, None

    if not raw_pkt.haslayer(IP):
        return None, None

    ip = raw_pkt[IP]
    ts = float(raw_pkt.time)

    # --- TCP ---
    if raw_pkt.haslayer(TCP):
        tcp = raw_pkt[TCP]
        header_len = (ip.ihl * 4) + (tcp.dataofs * 4)
        payload_len = max(len(raw_pkt) - header_len, 0)
        flags = int(tcp.flags)
        pkt = PacketRecord(
            timestamp   = ts,
            length      = int(len(raw_pkt)),
            direction   = "fwd",   # FlowManager resolves bidirectional
            tcp_flags   = flags,
            header_len  = header_len,
            push_flag   = bool(flags & 0x08),
            urg_flag    = bool(flags & 0x20),
            window_size = int(tcp.window),
        )
        key = FlowKey(
            src_ip=ip.src, dst_ip=ip.dst,
            src_port=int(tcp.sport), dst_port=int(tcp.dport),
            protocol=_PROTO_TCP,
        )
        return pkt, key

    # --- UDP ---
    if raw_pkt.haslayer(UDP):
        udp = raw_pkt[UDP]
        header_len = (ip.ihl * 4) + 8
        pkt = PacketRecord(
            timestamp   = ts,
            length      = int(len(raw_pkt)),
            direction   = "fwd",
            tcp_flags   = 0,
            header_len  = header_len,
            push_flag   = False,
            urg_flag    = False,
            window_size = 0,
        )
        key = FlowKey(
            src_ip=ip.src, dst_ip=ip.dst,
            src_port=int(udp.sport), dst_port=int(udp.dport),
            protocol=_PROTO_UDP,
        )
        return pkt, key

    # --- ICMP (no ports — use 0) ---
    if raw_pkt.haslayer(ICMP):
        header_len = (ip.ihl * 4) + 8
        pkt = PacketRecord(
            timestamp   = ts,
            length      = int(len(raw_pkt)),
            direction   = "fwd",
            tcp_flags   = 0,
            header_len  = header_len,
            push_flag   = False,
            urg_flag    = False,
            window_size = 0,
        )
        key = FlowKey(
            src_ip=ip.src, dst_ip=ip.dst,
            src_port=0, dst_port=0,
            protocol=_PROTO_ICMP,
        )
        return pkt, key

    return None, None
