"""
ORM models for the metadata DB (plan §6 + §15).

Portable across SQLite (dev) and Postgres (docker/prod): only generic types, JSON
instead of JSONB, and string enums instead of native DB enums.

Secrets (`*_encrypted` columns) are Fernet ciphertext — see app/utils/crypto.py.
Never read them directly; go through the repository/resolver which decrypts.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    type_annotation_map = {dict[str, Any]: JSON, list[Any]: JSON}


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )


# ── Datasources & schema catalog ──────────────────────────────────────────────


class Datasource(TimestampMixin, Base):
    """A target database users ask questions about."""

    __tablename__ = "datasources"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)
    dialect: Mapped[str] = mapped_column(String(30))  # postgresql | mysql | sqlite
    connection_url_encrypted: Mapped[str] = mapped_column(Text)
    schema_filter: Mapped[str | None] = mapped_column(String(120))  # e.g. "public"
    table_prefix: Mapped[str | None] = mapped_column(String(30))  # e.g. "tb_"
    excluded_patterns: Mapped[list[Any]] = mapped_column(default=list)
    include_sample_values: Mapped[bool] = mapped_column(Boolean, default=True)
    read_only_verified: Mapped[bool] = mapped_column(Boolean, default=False)
    schema_version: Mapped[int] = mapped_column(Integer, default=0)
    last_synced_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    tables: Mapped[list[CatalogTable]] = relationship(
        back_populates="datasource", cascade="all, delete-orphan"
    )


class CatalogTable(Base):
    __tablename__ = "catalog_tables"
    __table_args__ = (UniqueConstraint("datasource_id", "table_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    datasource_id: Mapped[int] = mapped_column(ForeignKey("datasources.id", ondelete="CASCADE"))
    table_name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text)
    is_hidden: Mapped[bool] = mapped_column(Boolean, default=False)
    row_count_est: Mapped[int | None] = mapped_column(Integer)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    datasource: Mapped[Datasource] = relationship(back_populates="tables")
    columns: Mapped[list[CatalogColumn]] = relationship(
        back_populates="table", cascade="all, delete-orphan", order_by="CatalogColumn.ordinal"
    )


class CatalogColumn(Base):
    __tablename__ = "catalog_columns"
    __table_args__ = (UniqueConstraint("table_id", "column_name"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    table_id: Mapped[int] = mapped_column(ForeignKey("catalog_tables.id", ondelete="CASCADE"))
    column_name: Mapped[str] = mapped_column(String(255))
    ordinal: Mapped[int] = mapped_column(Integer, default=0)
    data_type: Mapped[str] = mapped_column(String(255))
    nullable: Mapped[bool] = mapped_column(Boolean, default=True)
    is_pk: Mapped[bool] = mapped_column(Boolean, default=False)
    fk_table: Mapped[str | None] = mapped_column(String(255))
    fk_column: Mapped[str | None] = mapped_column(String(255))
    fk_source: Mapped[str | None] = mapped_column(String(20))  # introspected|annotated|inferred
    enum_values: Mapped[list[Any]] = mapped_column(default=list)
    sample_values: Mapped[list[Any]] = mapped_column(default=list)
    description: Mapped[str | None] = mapped_column(Text)
    is_sensitive: Mapped[bool] = mapped_column(Boolean, default=False)
    is_hidden: Mapped[bool] = mapped_column(Boolean, default=False)

    table: Mapped[CatalogTable] = relationship(back_populates="columns")


class Example(Base):
    """Verified question → SQL pair, used for few-shot retrieval (Phase 3)."""

    __tablename__ = "examples"

    id: Mapped[int] = mapped_column(primary_key=True)
    datasource_id: Mapped[int] = mapped_column(ForeignKey("datasources.id", ondelete="CASCADE"))
    question: Mapped[str] = mapped_column(Text)
    sql: Mapped[str] = mapped_column(Text)
    verified: Mapped[bool] = mapped_column(Boolean, default=False)
    source: Mapped[str] = mapped_column(String(20), default="manual")  # manual | feedback
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ── Query history & feedback ──────────────────────────────────────────────────


class QueryHistory(Base):
    __tablename__ = "query_history"

    id: Mapped[int] = mapped_column(primary_key=True)
    datasource_id: Mapped[int | None] = mapped_column(
        ForeignKey("datasources.id", ondelete="SET NULL")
    )
    question: Mapped[str] = mapped_column(Text)
    rewritten_question: Mapped[str | None] = mapped_column(Text)
    selected_tables: Mapped[list[Any]] = mapped_column(default=list)
    generated_sql: Mapped[str | None] = mapped_column(Text)
    final_sql: Mapped[str | None] = mapped_column(Text)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    abstained: Mapped[bool] = mapped_column(Boolean, default=False)
    error: Mapped[str | None] = mapped_column(Text)
    row_count: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[float] = mapped_column(Float, default=0.0)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0)
    model: Mapped[str | None] = mapped_column(String(255))
    trace: Mapped[dict[str, Any]] = mapped_column(default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(primary_key=True)
    query_id: Mapped[int] = mapped_column(ForeignKey("query_history.id", ondelete="CASCADE"))
    rating: Mapped[int] = mapped_column(Integer)  # +1 / -1
    corrected_sql: Mapped[str | None] = mapped_column(Text)
    comment: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ── Bring-your-own-model (plan §15) ───────────────────────────────────────────


class LLMProvider(TimestampMixin, Base):
    """A configured provider account: 'my OpenAI key', 'local Ollama', ..."""

    __tablename__ = "llm_providers"

    id: Mapped[int] = mapped_column(primary_key=True)
    provider_type: Mapped[str] = mapped_column(String(40))  # key in app/llm/registry.py
    display_name: Mapped[str] = mapped_column(String(120), unique=True)
    api_key_encrypted: Mapped[str | None] = mapped_column(Text)
    api_base: Mapped[str | None] = mapped_column(String(500))
    extra: Mapped[dict[str, Any]] = mapped_column(default=dict)  # api_version, region, ...
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    last_test_ok: Mapped[bool | None] = mapped_column(Boolean)
    last_tested_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))

    models: Mapped[list[LLMModel]] = relationship(
        back_populates="provider", cascade="all, delete-orphan"
    )


class LLMModel(Base):
    """A model enabled under a provider."""

    __tablename__ = "llm_models"
    __table_args__ = (UniqueConstraint("provider_id", "model_name", "kind"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    provider_id: Mapped[int] = mapped_column(ForeignKey("llm_providers.id", ondelete="CASCADE"))
    model_name: Mapped[str] = mapped_column(String(255))  # as the provider names it
    display_name: Mapped[str | None] = mapped_column(String(255))
    kind: Mapped[str] = mapped_column(String(20), default="chat")  # chat | embedding
    context_window: Mapped[int | None] = mapped_column(Integer)
    embedding_dim: Mapped[int | None] = mapped_column(Integer)
    temperature_default: Mapped[float] = mapped_column(Float, default=0.0)
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)

    provider: Mapped[LLMProvider] = relationship(back_populates="models")


class ModelAssignment(Base):
    """Which model serves which pipeline role (sql_generation, embedding, ...)."""

    __tablename__ = "model_assignments"

    role: Mapped[str] = mapped_column(String(40), primary_key=True)
    model_id: Mapped[int] = mapped_column(ForeignKey("llm_models.id", ondelete="CASCADE"))
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )

    model: Mapped[LLMModel] = relationship()
