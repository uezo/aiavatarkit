from types import MappingProxyType
from typing import Dict, Iterator, Mapping, Optional

from .models import AsteriskSessionData


class AsteriskCallRegistry:
    """Own live sessions and every ARI channel-to-session index.

    All mutations are synchronous so a caller can reserve or release a channel
    identity without yielding between the session update and reverse-index
    update.
    """

    def __init__(self) -> None:
        self._sessions: Dict[str, AsteriskSessionData] = {}
        self._caller_sessions: Dict[str, str] = {}
        self._media_sessions: Dict[str, str] = {}
        self._destination_sessions: Dict[str, str] = {}

    @property
    def sessions(self) -> Mapping[str, AsteriskSessionData]:
        return MappingProxyType(self._sessions)

    def __len__(self) -> int:
        return len(self._sessions)

    def __iter__(self) -> Iterator[str]:
        return iter(self._sessions)

    def get(self, session_id: str) -> Optional[AsteriskSessionData]:
        return self._sessions.get(session_id)

    def register(self, session: AsteriskSessionData) -> None:
        if session.session_id in self._sessions:
            raise ValueError(f"Session is already registered: {session.session_id}")
        self._validate_channel_available(
            self._caller_sessions,
            session.ari_caller_channel_id,
            session.session_id,
            role="caller",
        )
        if session.media_channel_id:
            self._validate_channel_available(
                self._media_sessions,
                session.media_channel_id,
                session.session_id,
                role="media",
            )
        if session.destination_channel_id:
            self._validate_channel_available(
                self._destination_sessions,
                session.destination_channel_id,
                session.session_id,
                role="destination",
            )

        self._sessions[session.session_id] = session
        self._bind_channel(
            self._caller_sessions,
            session.ari_caller_channel_id,
            session.session_id,
            role="caller",
        )
        if session.media_channel_id:
            self.bind_media(session, session.media_channel_id)
        if session.destination_channel_id:
            self.bind_destination(session, session.destination_channel_id)

    def remove(self, session_id: str) -> Optional[AsteriskSessionData]:
        session = self._sessions.pop(session_id, None)
        if session is None:
            return None
        self._unbind_if_owned(
            self._caller_sessions,
            session.ari_caller_channel_id,
            session_id,
        )
        self._unbind_if_owned(
            self._media_sessions,
            session.media_channel_id,
            session_id,
        )
        self._unbind_if_owned(
            self._destination_sessions,
            session.destination_channel_id,
            session_id,
        )
        return session

    def by_caller(self, channel_id: str) -> Optional[str]:
        return self._caller_sessions.get(channel_id)

    def by_media(self, channel_id: str) -> Optional[str]:
        return self._media_sessions.get(channel_id)

    def by_destination(self, channel_id: str) -> Optional[str]:
        return self._destination_sessions.get(channel_id)

    def bind_media(self, session: AsteriskSessionData, channel_id: str) -> None:
        self._require_registered(session)
        self._validate_channel_available(
            self._media_sessions,
            channel_id,
            session.session_id,
            role="media",
        )
        previous = session.media_channel_id
        if previous and previous != channel_id:
            self._unbind_if_owned(
                self._media_sessions,
                previous,
                session.session_id,
            )
        self._bind_channel(
            self._media_sessions,
            channel_id,
            session.session_id,
            role="media",
        )
        session.media_channel_id = channel_id

    def unbind_media(
        self,
        session: AsteriskSessionData,
        channel_id: Optional[str] = None,
    ) -> None:
        target = channel_id or session.media_channel_id
        self._unbind_if_owned(
            self._media_sessions,
            target,
            session.session_id,
        )
        if session.media_channel_id == target:
            session.media_channel_id = ""

    def bind_destination(
        self,
        session: AsteriskSessionData,
        channel_id: str,
    ) -> None:
        self._require_registered(session)
        self._validate_channel_available(
            self._destination_sessions,
            channel_id,
            session.session_id,
            role="destination",
        )
        previous = session.destination_channel_id
        if previous and previous != channel_id:
            self._unbind_if_owned(
                self._destination_sessions,
                previous,
                session.session_id,
            )
        self._bind_channel(
            self._destination_sessions,
            channel_id,
            session.session_id,
            role="destination",
        )
        session.destination_channel_id = channel_id

    def unbind_destination(
        self,
        session: AsteriskSessionData,
        channel_id: Optional[str] = None,
    ) -> None:
        target = channel_id or session.destination_channel_id
        self._unbind_if_owned(
            self._destination_sessions,
            target,
            session.session_id,
        )
        if session.destination_channel_id == target:
            session.destination_channel_id = ""

    def _require_registered(self, session: AsteriskSessionData) -> None:
        if self._sessions.get(session.session_id) is not session:
            raise ValueError(f"Session is not registered: {session.session_id}")

    @staticmethod
    def _validate_channel_available(
        index: Dict[str, str],
        channel_id: str,
        session_id: str,
        *,
        role: str,
    ) -> None:
        if not channel_id:
            raise ValueError(f"{role} channel ID is required")
        owner = index.get(channel_id)
        if owner is not None and owner != session_id:
            raise ValueError(
                f"{role} channel {channel_id!r} belongs to another session"
            )

    @classmethod
    def _bind_channel(
        cls,
        index: Dict[str, str],
        channel_id: str,
        session_id: str,
        *,
        role: str,
    ) -> None:
        cls._validate_channel_available(
            index,
            channel_id,
            session_id,
            role=role,
        )
        index[channel_id] = session_id

    @staticmethod
    def _unbind_if_owned(
        index: Dict[str, str],
        channel_id: str,
        session_id: str,
    ) -> None:
        if channel_id and index.get(channel_id) == session_id:
            index.pop(channel_id, None)
