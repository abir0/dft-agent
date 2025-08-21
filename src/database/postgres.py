import os
from datetime import datetime
from typing import List, Optional, Type, TypeVar, Union

from dotenv import load_dotenv
from pydantic import BaseModel
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Table,
    select,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker

load_dotenv()

Base = declarative_base()
T = TypeVar("T", bound=BaseModel)


class AsyncPostgresManager:
    """
    Asynchronous CRUD manager for PostgreSQL using SQLAlchemy.

    This class provides methods to create, read, update, delete, and query data
    in any PostgreSQL table using SQLAlchemy. All operations are implemented asynchronously.
    """

    def __init__(self, url: Optional[str] = None):
        self.engine = create_async_engine(
            url
            or os.getenv(
                "DATABASE_URL", "postgresql+asyncpg://user:password@localhost/dbname"
            ),
            echo=False,
        )
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )

        self.tables = {}  # Store table definitions

    async def init_db(self):
        """Initialize the database by creating all tables."""
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        await self.engine.dispose()

    async def create_table_from_pydantic(self, model: Type[BaseModel], table_name: str):
        """
        Create a new table from a Pydantic model.

        Args:
            model: Pydantic model class
            table_name: Name for the table
        """
        if table_name in self.tables:
            return

        # Get model fields and their types
        model_fields = model.model_fields

        # Map Pydantic types to SQLAlchemy types
        type_mapping = {
            int: Integer,
            str: String,
            float: Float,
            bool: Boolean,
            datetime: DateTime,
            dict: JSON,
            list: JSON,
        }

        columns = []
        for field_name, field_info in model_fields.items():
            # Get the field type
            field_type = field_info.annotation

            # Handle Optional types
            if getattr(field_type, "__origin__", None) is Union:
                field_type = field_type.__args__[0]

            # Handle nested Pydantic models and lists
            if (
                (isinstance(field_type, type) and issubclass(field_type, BaseModel))
                or field_type in (dict, list)
                or getattr(field_type, "__origin__", None) is list
            ):
                sql_type = JSON
            else:
                # Map to SQLAlchemy type
                sql_type = type_mapping.get(field_type, String)

            # Create column
            is_primary = field_name == "id"
            column = Column(field_name, sql_type, primary_key=is_primary)
            columns.append(column)

        # Create table definition
        table = Table(table_name, Base.metadata, *columns)
        self.tables[table_name] = (table, model)

        # Create table in database
        async with self.engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

    async def create_data(self, table_name: str, data: dict) -> dict:
        """
        Create a new data entry in the specified table.
        If the ID already exists, update the existing record.

        Args:
            table_name: The target table name
            data: The data to be created
        """
        if table_name not in self.tables:
            raise ValueError(
                f"Table '{table_name}' not found. Create it first using create_table_from_pydantic()"
            )

        table, model = self.tables[table_name]

        # Validate data against Pydantic model
        validated_data = model(**data).model_dump()

        async with self.async_session() as session:
            try:
                # Create an "upsert" statement using PostgreSQL-specific insert
                stmt = pg_insert(table).values(**validated_data)
                stmt = stmt.on_conflict_do_update(
                    index_elements=["id"], set_=validated_data
                )
                await session.execute(stmt)
                await session.commit()
                return validated_data
            except Exception as e:
                await session.rollback()
                raise e

    async def read_data(self, table_name: str, data_id: str) -> Optional[dict]:
        """Read a data entry from the specified table."""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")

        table, _ = self.tables[table_name]

        async with self.async_session() as session:
            stmt = select(table).where(table.c.id == data_id)
            result = await session.execute(stmt)
            row = result.first()
            return row.data if row else None

    async def read_all_data(
        self, table_name: str, max_item_count: int = 10
    ) -> List[dict]:
        """Read all data entries in the specified table."""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")

        table, _ = self.tables[table_name]

        async with self.async_session() as session:
            stmt = select(table).limit(max_item_count)
            result = await session.execute(stmt)
            rows = result.all()
            # Convert each row to a dictionary
            return [dict(row._mapping) for row in rows]

    async def update_data(self, table_name: str, data: dict) -> dict:
        """Update a data entry in the specified table."""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")

        table, model = self.tables[table_name]
        validated_data = model(**data).model_dump()

        async with self.async_session() as session:
            stmt = (
                table.update()
                .where(table.c.id == data.get("id"))
                .values(data=validated_data, updated_at=datetime.utcnow())
            )
            result = await session.execute(stmt)
            if result.rowcount == 0:
                raise ValueError(
                    f"Data with id '{data.get('id')}' not found in table '{table_name}'"
                )
            await session.commit()
            return validated_data

    async def delete_data(self, table_name: str, data_id: str) -> None:
        """Delete a data entry from the specified table."""
        if table_name not in self.tables:
            raise ValueError(f"Table '{table_name}' not found")

        table, _ = self.tables[table_name]

        async with self.async_session() as session:
            stmt = table.delete().where(table.c.id == data_id)
            result = await session.execute(stmt)
            if result.rowcount == 0:
                print(f"Data with id '{data_id}' not found in table '{table_name}'")
            await session.commit()

    async def query_data(
        self, query: str, parameters: Optional[List[dict]] = None
    ) -> List[dict]:
        """
        Query data using SQL expressions.

        Args:
            query: The SQL query string
            parameters: Optional parameters for the query as a list of dictionaries

        Returns:
            A list of data entries matching the query
        """
        async with self.async_session() as session:
            # Execute the query directly if no parameters
            if not parameters:
                result = await session.execute(text(query))
            else:
                # Execute with parameters if provided
                result = await session.execute(text(query), parameters[0])

            return [dict(row._mapping) for row in result]
