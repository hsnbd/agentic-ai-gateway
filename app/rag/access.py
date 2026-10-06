"""Who may see and change which RAG collections.

A collection created with a virtual key belongs to that key's team, or to the
key itself when it has no team. Collections with no owner (created by the
master key or a console admin, or before ownership existed) are global:
every caller can read and search them, but only operators can change them.

Collections a caller cannot read are reported as not found, so their
existence is not revealed to other tenants.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sqlalchemy import ColumnElement, and_, or_, true

from app.db.models import RagCollection

if TYPE_CHECKING:
    from app.api.deps import GatewayPrincipal
    from app.core.pipeline import RequestContext


@dataclass(frozen=True)
class RagAccess:
    #: The master key and console admins: see and change everything.
    unrestricted: bool = False
    #: Console viewers: read everything (writes are refused by the route guard).
    read_all: bool = False
    team_id: str | None = None
    key_id: str | None = None

    @classmethod
    def for_principal(cls, principal: GatewayPrincipal) -> RagAccess:
        if principal.role == "admin":
            return UNRESTRICTED
        if principal.kind == "console":
            return cls(read_all=True)
        return cls(team_id=principal.team_id, key_id=principal.identifier)

    @classmethod
    def for_context(cls, ctx: RequestContext) -> RagAccess:
        """The access of a chat request's caller, as resolved by the auth stage."""
        if ctx.key_id in (None, "master"):
            return UNRESTRICTED
        return cls(team_id=ctx.team_id, key_id=ctx.key_id)

    def owner_fields(self) -> dict[str, str | None]:
        """Ownership for a collection this caller creates."""
        if self.unrestricted or self.read_all:
            return {"owner_team_id": None, "owner_key_id": None}
        if self.team_id:
            return {"owner_team_id": self.team_id, "owner_key_id": None}
        return {"owner_team_id": None, "owner_key_id": self.key_id}

    def _owns(self, collection: RagCollection) -> bool:
        return bool(
            (self.team_id and collection.owner_team_id == self.team_id)
            or (self.key_id and collection.owner_key_id == self.key_id)
        )

    def can_read(self, collection: RagCollection) -> bool:
        if self.unrestricted or self.read_all:
            return True
        is_global = collection.owner_team_id is None and collection.owner_key_id is None
        return is_global or self._owns(collection)

    def can_write(self, collection: RagCollection) -> bool:
        return self.unrestricted or self._owns(collection)

    def visible_filter(self) -> Any:
        """SQL condition selecting the collections this caller can read."""
        if self.unrestricted or self.read_all:
            return true()
        # Every restricted caller is a virtual key, so key_id is always set here.
        conditions: list[ColumnElement[bool]] = [
            and_(RagCollection.owner_team_id.is_(None), RagCollection.owner_key_id.is_(None)),
            RagCollection.owner_key_id == self.key_id,
        ]
        if self.team_id:
            conditions.append(RagCollection.owner_team_id == self.team_id)
        return or_(*conditions)


UNRESTRICTED = RagAccess(unrestricted=True)
