"""
flows/flow_manager.py
Thread-safe flow table — tracks active flows and exports completed ones.

Responsibilities
----------------
  - Maintain a dict[FlowKey -> NetworkFlow] of active flows
  - Route incoming packets to the correct flow (or create a new one)
  - Detect flow completion via TCP FIN/RST flags
  - Detect flow timeout (idle: 120s, absolute: 600s)
  - Export completed flows to a callback queue for feature extraction

Usage
-----
    from flows.flow_manager import FlowManager

    def on_flow_ready(flow: NetworkFlow):
        features = extract_features(flow)
        result   = predict_flow(features)
        ...

    mgr = FlowManager(on_flow_complete=on_flow_ready)
    mgr.process_packet(pkt_record, flow_key)

    # Call periodically to expire timed-out flows:
    mgr.flush_expired()
"""

import logging
import threading
import time
from typing import Callable, Dict

from flows.flow import FlowKey, NetworkFlow, PacketRecord

logger = logging.getLogger(__name__)

# TCP flag bit masks
_TCP_FIN = 0x01
_TCP_RST = 0x04


class FlowManager:
    """
    Thread-safe active flow table.

    Parameters
    ----------
    on_flow_complete : Callable[[NetworkFlow], None]
        Callback invoked when a flow is exported (completed or timed out).
        Called from the same thread that triggered the export.
    idle_timeout     : float, default 120s
        Export flow if no packets seen for this many seconds.
    absolute_timeout : float, default 600s
        Export flow regardless of activity after this many seconds.
    """

    def __init__(
        self,
        on_flow_complete: Callable[[NetworkFlow], None],
        idle_timeout:     float = 120.0,
        absolute_timeout: float = 600.0,
    ) -> None:
        self._flows:    Dict[FlowKey, NetworkFlow] = {}
        self._lock:     threading.RLock            = threading.RLock()
        self._on_complete                          = on_flow_complete
        self._idle_timeout                         = idle_timeout
        self._absolute_timeout                     = absolute_timeout

    def process_packet(self, pkt: PacketRecord, key: FlowKey) -> None:
        with self._lock:
            flow = self._flows.get(key)

            if flow is None:
                reverse_key = FlowKey(
                    src_ip=key.dst_ip, dst_ip=key.src_ip,
                    src_port=key.dst_port, dst_port=key.src_port,
                    protocol=key.protocol,
                )
                flow = self._flows.get(reverse_key)
                if flow is not None:
                    pkt_reversed = PacketRecord(
                        timestamp=pkt.timestamp, length=pkt.length,
                        direction="bwd",
                        tcp_flags=pkt.tcp_flags,    header_len=pkt.header_len,
                        push_flag=pkt.push_flag,    urg_flag=pkt.urg_flag,
                        window_size=pkt.window_size,
                    )
                    flow.add_packet(pkt_reversed)
                    key = reverse_key
                    logger.debug("[FlowManager] Existing flow (reverse) | pkts=%d | active_flows=%d", flow.total_packets, len(self._flows))
                else:
                    flow = NetworkFlow(key=key)
                    self._flows[key] = flow
                    flow.add_packet(pkt)
                    logger.debug("[FlowManager] New flow: %s | active_flows=%d", key, len(self._flows))
                    return
            else:
                flow.add_packet(pkt)
                logger.debug("[FlowManager] Existing flow (fwd) | pkts=%d | active_flows=%d", flow.total_packets, len(self._flows))

            # Export on TCP FIN or RST
            if pkt.tcp_flags & (_TCP_FIN | _TCP_RST):
                flow.mark_finished()
                self._flows.pop(key, None)
                export_flow = flow
            else:
                export_flow = None

        if export_flow is not None:
            self._export(key, export_flow)

    def flush_expired(self) -> int:
        expired_keys = []
        with self._lock:
            active_count = len(self._flows)
            for key, flow in self._flows.items():
                if flow.is_idle(self._idle_timeout) or flow.is_expired(self._absolute_timeout):
                    expired_keys.append(key)

        logger.info(
            "[FlushCycle] active_flows=%d  expired_this_cycle=%d",
            active_count, len(expired_keys),
        )

        exported_flows = []
        with self._lock:
            for key in expired_keys:
                flow = self._flows.pop(key, None)
                if flow is not None:
                    exported_flows.append((key, flow))

        for key, flow in exported_flows:
            self._export(key, flow)

        if exported_flows:
            logger.info("Flushed %d expired flows", len(exported_flows))
        return len(exported_flows)

    def flush_all(self) -> int:
        with self._lock:
            flows_to_export = list(self._flows.items())
            self._flows.clear()

        for key, flow in flows_to_export:
            self._export(key, flow)

        logger.info("Shutdown flush: exported %d flows", len(flows_to_export))
        return len(flows_to_export)

    @property
    def active_flow_count(self) -> int:
        with self._lock:
            return len(self._flows)

    def _export(self, key: FlowKey, flow: NetworkFlow) -> None:
        """Remove flow from table and invoke the completion callback."""
        reason = "FIN/RST" if flow.is_finished else (
            "idle_timeout" if flow.is_idle(self._idle_timeout) else "absolute_timeout"
        )
        logger.info(
            "[Export] reason=%s | %s | pkts=%d | bytes=%d | duration_ms=%.0f",
            reason, key, flow.total_packets, flow.total_bytes,
            flow.duration_us / 1000,
        )
        try:
            self._on_complete(flow)
        except Exception as exc:
            logger.exception("on_flow_complete callback failed: %s", exc)
