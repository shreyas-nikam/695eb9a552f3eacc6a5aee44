
import streamlit as st
import asyncio
import uuid
from datetime import datetime
import time
from typing import Optional, List, Dict, Any, AsyncGenerator

# SQL Alchemy imports
from sqlalchemy import select, func, text, DateTime, String, Float, JSON, ForeignKey, Index
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import declarative_base, Mapped, mapped_column, sessionmaker, relationship, selectinload
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine

import sys
import json  # For Redis serialization
import random  # For sample scores

# Helper function to run async code in Streamlit


def run_async(coro):
    """Helper to run async code in Streamlit without closing the event loop."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


# --- Configuration ---
# FIX: Changed DATABASE_URL back to "sqlite+aiosqlite" as create_async_engine requires an async driver.
# The "aiosqlite" driver (from the 'aiosqlite' package) is necessary for SQLAlchemy's asyncio extension
# to work with SQLite. The previous change to "sqlite" caused 'pysqlite' (a synchronous driver) to be used,
# leading to "InvalidRequestError: The asyncio extension requires an async driver to be used."
# Ensure 'aiosqlite' is installed (e.g., `pip install aiosqlite`).
# In-memory SQLite for demonstration, usually PostgreSQL
DATABASE_URL = "sqlite+aiosqlite:///./test.db"

# --- SQLAlchemy Base and Mixins ---
Base = declarative_base()


class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)

# --- Models ---


class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    occupation_code: Mapped[Optional[str]] = mapped_column(String(20))
    education_level: Mapped[Optional[str]] = mapped_column(String(50))
    years_experience: Mapped[Optional[float]] = mapped_column(Float)
    assessments: Mapped[List["Assessment"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin")
    scores: Mapped[List["AIRScore"]] = relationship(
        back_populates="user", cascade="all, delete-orphan", lazy="selectin")

    def to_dict(self):
        return {
            "id": self.id,
            "email": self.email,
            "name": self.name,
            "occupation_code": self.occupation_code,
            "education_level": self.education_level,
            "years_experience": self.years_experience,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class Assessment(Base, TimestampMixin):
    __tablename__ = "assessments"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
    component: Mapped[str] = mapped_column(String(50))
    current_ability: Mapped[float] = mapped_column(Float, default=0.0)
    items_administered: Mapped[int] = mapped_column(default=0)
    user: Mapped["User"] = relationship(back_populates="assessments")


class AIRScore(Base, TimestampMixin):
    __tablename__ = "air_scores"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    occupation_code: Mapped[str] = mapped_column(String(20))
    air_score: Mapped[float] = mapped_column(Float)
    vr_score: Mapped[float] = mapped_column(Float)
    hr_score: Mapped[float] = mapped_column(Float)
    synergy_score: Mapped[float] = mapped_column(Float)
    ci_lower: Mapped[float] = mapped_column(Float)
    ci_upper: Mapped[float] = mapped_column(Float)
    parameter_version: Mapped[str] = mapped_column(String(20))
    calculation_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    user: Mapped["User"] = relationship(back_populates="scores")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "occupation_code": self.occupation_code,
            "air_score": self.air_score,
            "vr_score": self.vr_score,
            "hr_score": self.hr_score,
            "synergy_score": self.synergy_score,
            "ci_lower": self.ci_lower,
            "ci_upper": self.ci_upper,
            "parameter_version": self.parameter_version,
            "calculation_metadata": self.calculation_metadata,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class DomainEvent(Base, TimestampMixin):
    __tablename__ = "domain_events"
    id: Mapped[str] = mapped_column(
        String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type: Mapped[str] = mapped_column(
        String(100), nullable=False, index=True)
    aggregate_type: Mapped[str] = mapped_column(String(100), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(
        String(36), nullable=False, index=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    published_at: Mapped[Optional[datetime]
                         ] = mapped_column(DateTime(timezone=True))
    __table_args__ = (
        Index('ix_events_status_created', 'status', 'created_at'),)


# --- Database Engine and Session Setup ---
async_engine = create_async_engine(
    DATABASE_URL, echo=False, future=True, pool_size=10, max_overflow=20)
AsyncSessionLocal = sessionmaker(
    async_engine, expire_on_commit=False, class_=AsyncSession
)


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session

# --- Repositories ---


class BaseRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def create(self, model_instance: Base):
        self.session.add(model_instance)
        await self.session.flush()  # Flush to get ID if needed
        return model_instance

    async def get_by_id(self, model_id: str, model_class: type[Base]):
        result = await self.session.execute(
            select(model_class).filter_by(id=model_id)
        )
        return result.scalars().first()


class UserRepository(BaseRepository):
    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_by_id(self, user_id: str) -> Optional[User]:
        return await super().get_by_id(user_id, User)

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).filter_by(email=email)
        )
        return result.scalars().first()

    async def get_user_with_scores_eager(self, user_id: str) -> Optional[User]:
        # Corrected to use selectinload for eager loading of collections
        result = await self.session.execute(
            select(User).filter_by(id=user_id).options(
                selectinload(User.scores))
        )
        return result.scalars().unique().first()

    async def get_latest_score(self, user_id: str) -> Optional[AIRScore]:
        result = await self.session.execute(
            select(AIRScore).filter_by(user_id=user_id).order_by(
                AIRScore.created_at.desc()).limit(1)
        )
        return result.scalars().first()


class AIRScoreRepository(BaseRepository):
    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_by_id(self, score_id: str) -> Optional[AIRScore]:
        return await super().get_by_id(score_id, AIRScore)


class DomainEventRepository(BaseRepository):
    def __init__(self, session: AsyncSession):
        super().__init__(session)

    async def get_pending_events(self, limit: int = 10) -> List[DomainEvent]:
        result = await self.session.execute(
            select(DomainEvent).filter_by(status="pending").order_by(
                DomainEvent.created_at).limit(limit)
        )
        return result.scalars().all()

    async def mark_as_published(self, event_ids: List[str]):
        # Operates on the session passed during initialization
        for event_id in event_ids:
            result = await self.session.execute(select(DomainEvent).filter_by(id=event_id))
            event = result.scalars().first()
            if event:
                event.status = "published"
                event.published_at = datetime.utcnow()
                self.session.add(event)
        await self.session.commit()

# --- Caching with Redis ---


class CachedUserRepository(UserRepository):
    def __init__(self, session: AsyncSession, redis_client):
        super().__init__(session)
        self.redis_client = redis_client
        self.USER_CACHE_PREFIX = "user:"
        self.LATEST_SCORE_CACHE_PREFIX = "latest_airscore:"
        self.CACHE_TTL = 3600  # 1 hour

    async def _get_from_cache(self, key: str) -> Optional[Dict]:
        data = await self.redis_client.get(key)
        return json.loads(data) if data else None

    async def _set_to_cache(self, key: str, data: Dict):
        await self.redis_client.set(key, json.dumps(data), ex=self.CACHE_TTL)

    async def get_by_id_cached(self, user_id: str) -> Optional[User]:
        cache_key = f"{self.USER_CACHE_PREFIX}{user_id}"
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            # Reconstruct User from dict, handling datetime objects
            if 'created_at' in cached_data and cached_data['created_at']:
                cached_data['created_at'] = datetime.fromisoformat(
                    cached_data['created_at'])
            if 'updated_at' in cached_data and cached_data['updated_at']:
                cached_data['updated_at'] = datetime.fromisoformat(
                    cached_data['updated_at'])
            return User(**cached_data)

        user = await super().get_by_id(user_id)
        if user:
            await self._set_to_cache(cache_key, user.to_dict())
        return user

    async def get_latest_score_cached(self, user_id: str) -> Optional[AIRScore]:
        cache_key = f"{self.LATEST_SCORE_CACHE_PREFIX}{user_id}"
        cached_data = await self._get_from_cache(cache_key)
        if cached_data:
            # Reconstruct AIRScore from dict, handling datetime objects
            if 'created_at' in cached_data and cached_data['created_at']:
                cached_data['created_at'] = datetime.fromisoformat(
                    cached_data['created_at'])
            if 'updated_at' in cached_data and cached_data['updated_at']:
                cached_data['updated_at'] = datetime.fromisoformat(
                    cached_data['updated_at'])
            return AIRScore(**cached_data)

        score = await super().get_latest_score(user_id)
        if score:
            await self._set_to_cache(cache_key, score.to_dict())
        return score

    async def invalidate_user_cache(self, user_id: str):
        await self.redis_client.delete(f"{self.USER_CACHE_PREFIX}{user_id}")
        await self.redis_client.delete(f"{self.LATEST_SCORE_CACHE_PREFIX}{user_id}")

# --- Helper functions ---


async def init_db():
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def run_db_setup_and_create_user():
    await init_db()
    async for session in get_session():
        user_repo = UserRepository(session)
        # Clean up existing data for repeatable demo
        await session.execute(text("DELETE FROM domain_events"))
        await session.execute(text("DELETE FROM air_scores"))
        await session.execute(text("DELETE FROM assessments"))
        await session.execute(text("DELETE FROM users"))
        await session.commit()

        alex = User(email="alex.smith@innovateai.com", name="Alex Smith",
                    occupation_code="SWE", education_level="Masters", years_experience=5.5)
        jane = User(email="jane.doe@innovateai.com", name="Jane Doe",
                    occupation_code="DS", education_level="PhD", years_experience=3.0)

        await user_repo.create(alex)
        await user_repo.create(jane)
        await session.commit()
        await session.refresh(alex)
        await session.refresh(jane)
        return alex, jane


async def create_sample_airscore(user_id: str, occupation_code: str, session: AsyncSession) -> AIRScore:
    score = AIRScore(
        user_id=user_id,
        occupation_code=occupation_code,
        air_score=float(f"{random.uniform(60, 95):.2f}"),
        vr_score=float(f"{random.uniform(50, 90):.2f}"),
        hr_score=float(f"{random.uniform(70, 99):.2f}"),
        synergy_score=float(f"{random.uniform(65, 95):.2f}"),
        ci_lower=float(f"{random.uniform(55, 75):.2f}"),
        ci_upper=float(f"{random.uniform(85, 99):.2f}"),
        parameter_version="v1.2.3",
        calculation_metadata={"model": "deep-ai-v3",
                              "date": datetime.now().isoformat()}
    )
    session.add(score)
    await session.commit()  # Commit here for atomic score creation
    await session.refresh(score)
    return score


async def calculate_and_store_airscore(user_id: str, occupation_code: str) -> tuple[AIRScore, DomainEvent]:
    async for session in get_session():
        # Create AIRScore
        score = AIRScore(
            user_id=user_id,
            occupation_code=occupation_code,
            air_score=float(f"{random.uniform(60, 95):.2f}"),
            vr_score=float(f"{random.uniform(50, 90):.2f}"),
            hr_score=float(f"{random.uniform(70, 99):.2f}"),
            synergy_score=float(f"{random.uniform(65, 95):.2f}"),
            ci_lower=float(f"{random.uniform(55, 75):.2f}"),
            ci_upper=float(f"{random.uniform(85, 99):.2f}"),
            parameter_version="v1.2.3",
            calculation_metadata={"model": "deep-ai-v3",
                                  "date": datetime.now().isoformat()}
        )
        session.add(score)

        # Create DomainEvent
        await session.flush()  # Ensure score has an ID before event payload
        event_payload = {
            "user_id": user_id,
            "air_score_id": score.id,
            "occupation_code": score.occupation_code,
            "air_score_value": score.air_score
        }
        event = DomainEvent(
            event_type="AIRScoreCalculated",
            aggregate_type="AIRScore",
            aggregate_id=score.id,
            payload=event_payload,
            status="pending"
        )
        session.add(event)

        await session.commit()
        await session.refresh(score)
        await session.refresh(event)
        return score, event

# Configure Page
st.set_page_config(
    page_title="QuLab: Data Architecture & Persistence", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Data Architecture & Persistence")
st.divider()

# --- Setup for Database and Redis ---

# Redis Setup
if "redis_status" not in st.session_state:
    st.session_state.redis_status = "Initializing..."

try:
    import redis.asyncio as aioredis
    # Attempt to connect to local Redis
    REDIS_CLIENT = aioredis.from_url(
        "redis://localhost:6379/0", encoding="utf-8", decode_responses=True)

    async def check_redis_connection():
        # Using a small timeout to avoid long waits if Redis is truly down
        await asyncio.wait_for(REDIS_CLIENT.ping(), timeout=1)

    try:
        run_async(check_redis_connection())
        st.session_state.redis_status = "Connected to local Redis."
    except Exception:
        # If connection fails, fall through to mock client
        st.session_state.redis_status = "Using Mock Redis Client (Local Redis unavailable)."
        from unittest import mock
        REDIS_CLIENT = mock.AsyncMock(spec=aioredis.Redis)
        REDIS_CLIENT.get.return_value = None
        REDIS_CLIENT.set.return_value = None
        REDIS_CLIENT.delete.return_value = None

except Exception as e:  # Catch ModuleNotFoundError for aioredis or any other unexpected error
    st.session_state.redis_status = f"Using Mock Redis Client (Redis module unavailable or failed to import)."
    # Ensure REDIS_CLIENT is always defined, even if aioredis itself can't be imported
    from unittest import mock
    # Use MagicMock as spec if aioredis.Redis is not available, this ensures the mock has basic async behavior
    REDIS_CLIENT = mock.AsyncMock(spec=mock.MagicMock())
    REDIS_CLIENT.get.return_value = None
    REDIS_CLIENT.set.return_value = None
    REDIS_CLIENT.delete.return_value = None

# --- Session State Initialization ---

if "current_page" not in st.session_state:
    st.session_state.current_page = "Introduction"
if "db_initialized" not in st.session_state:
    st.session_state.db_initialized = False
if "user_alex_id" not in st.session_state:
    st.session_state.user_alex_id = None
if "user_jane_id" not in st.session_state:
    st.session_state.user_jane_id = None
if "created_user_id" not in st.session_state:
    st.session_state.created_user_id = None
if "retrieved_user_with_scores" not in st.session_state:
    st.session_state.retrieved_user_with_scores = None
if "user_for_caching_id" not in st.session_state:
    st.session_state.user_for_caching_id = None
if "latest_airscore_id" not in st.session_state:
    st.session_state.latest_airscore_id = None
if "cache_hits" not in st.session_state:
    st.session_state.cache_hits = 0
if "cache_misses" not in st.session_state:
    st.session_state.cache_misses = 0
if "event_user_id" not in st.session_state:
    st.session_state.event_user_id = None
if "event_publisher_running" not in st.session_state:
    st.session_state.event_publisher_running = False
if "pending_events_display" not in st.session_state:
    st.session_state.pending_events_display = []
if "processed_events_display" not in st.session_state:
    st.session_state.processed_events_display = []
if "publisher_stop_event" not in st.session_state:
    st.session_state.publisher_stop_event = asyncio.Event()

# --- Sidebar Navigation ---

st.sidebar.title("Lab 2: Data Architecture")
options = [
    "Introduction",
    "1. Data Models",
    "2. DB Connectivity & Pooling",
    "3. Repository Pattern & N+1",
    "4. Caching with Redis",
    "5. Eventing (Outbox Pattern)"
]

# Use index to set default selection
current_index = options.index(
    st.session_state.current_page) if st.session_state.current_page in options else 0
selected_page = st.sidebar.selectbox(
    "Navigate Sections",
    options,
    index=current_index
)
st.session_state.current_page = selected_page

st.sidebar.markdown(f"**Redis Status:** {st.session_state.redis_status}")

# --- Page Content ---

if st.session_state.current_page == "Introduction":
    st.header("Introduction: Scaling the AI Backend's Data Layer")

    st.markdown(
        f"**Persona:** Alex, Senior Software Engineer at InnovateAI Solutions.")
    st.markdown(f"**Organization:** InnovateAI Solutions is a cutting-edge company building an AI-powered assessment platform. This platform helps users evaluate their skills and receive AI-driven recommendations.")

    st.markdown(f"**The Challenge:** In Lab 1, Alex successfully laid the foundation for a scalable Python backend. Now, the focus shifts to the critical data layer. As InnovateAI's platform gains traction, Alex faces the challenge of designing and implementing a robust, performant, and reliable data architecture. This involves not only persisting complex AI-related data but also ensuring efficient access patterns, handling concurrent requests, and reliably communicating events across a growing microservices ecosystem. Alex needs to ensure the data layer can support high throughput, low latency, and maintain data integrity, all while being adaptable to future changes.")

    st.markdown(f"This application simulates Alex's workflow in tackling these challenges, demonstrating practical application of modern data persistence patterns using SQLAlchemy 2.0 and Redis.")

    st.subheader("Key Objectives")
    st.markdown(
        f"- **Remember**: List SQLAlchemy relationship types and Redis data structures.")
    st.markdown(
        f"- **Understand**: Explain async database patterns and connection pooling.")
    st.markdown(
        f"- **Apply**: Implement repository pattern with SQLAlchemy 2.0.")
    st.markdown(
        f"- **Analyze**: Compare caching strategies for different access patterns.")
    st.markdown(f"- **Create**: Design event tables for pub/sub architecture.")

    st.subheader("Tools Introduced")
    st.markdown(
        f"- **PostgreSQL**: Primary database (ACID, JSON support, reliability)")
    st.markdown(f"- **SQLAlchemy 2.0**: ORM (Async support, type hints)")
    st.markdown(f"- **Alembic**: Migrations (Version control for schema)")
    st.markdown(f"- **Redis**: Cache + Pub/Sub (Speed, event distribution)")
    st.markdown(f"- **asyncpg**: Async driver (High-performance async)")

    st.subheader("Key Concepts")
    st.markdown(f"- Async database sessions with context managers")
    st.markdown(f"- Repository pattern for data access abstraction")
    st.markdown(f"- Connection pooling for scalability")
    st.markdown(f"- Event sourcing tables for pub/sub (Outbox pattern)")

elif st.session_state.current_page == "1. Data Models":
    st.title("1. Defining the Core Data Schema with SQLAlchemy 2.0")

    st.markdown(f"Alex starts by meticulously defining the data models that will underpin InnovateAI's AI assessment platform. This involves capturing user profiles, their ongoing assessments, the crucial AI-R scores, and a mechanism for tracking system events. He leverages SQLAlchemy 2.0's modern declarative mapping and type hints for clarity and robustness.")

    st.markdown(f"Correctly structured data models are fundamental for reliable data storage, efficient querying, and understanding the relationships between different entities in a complex AI system. SQLAlchemy's ORM helps bridge the gap between Python objects and relational database tables, ensuring type safety and reducing boilerplate code.")

    st.markdown(r"$$ \pi_{\text{Assessment.*}}(\text{Users} \bowtie_{\text{Users.id} = \text{Assessments.user\_id}} \text{Assessments}) $$ Relational Algebra is the foundation for database operations. A relationship between two tables, say `Users` and `Assessments`, implies a join operation. For example, to find all assessments for a user, one might perform a projection and join operation as shown above.")
    st.markdown(r"$$ \pi $$ denotes projection and $$ \bowtie $$ denotes natural join. SQLAlchemy relationships abstract this, making it object-oriented and intuitive in Python code.")

    st.subheader("SQLAlchemy Models Overview")
    st.markdown(
        f"Here's a glimpse into the SQLAlchemy model definitions Alex has created:")

    st.markdown(f"**User Model**: Represents user profiles.")
    st.code("""
class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    occupation_code: Mapped[Optional[str]] = mapped_column(String(20))
    education_level: Mapped[Optional[str]] = mapped_column(String(50))
    years_experience: Mapped[Optional[float]] = mapped_column(Float)
    assessments: Mapped[List["Assessment"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    scores: Mapped[List["AIRScore"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    """, language="python")

    st.markdown(f"**Assessment Model**: Tracks user's evaluation sessions.")
    st.code("""
class Assessment(Base, TimestampMixin):
    __tablename__ = "assessments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
    component: Mapped[str] = mapped_column(String(50))
    current_ability: Mapped[float] = mapped_column(Float, default=0.0)
    items_administered: Mapped[int] = mapped_column(default=0)
    user: Mapped["User"] = relationship(back_populates="assessments")
    """, language="python")

    st.markdown(f"**AIRScore Model**: Stores AI-generated assessment results.")
    st.code("""
class AIRScore(Base, TimestampMixin):
    __tablename__ = "air_scores"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    occupation_code: Mapped[str] = mapped_column(String(20))
    air_score: Mapped[float] = mapped_column(Float)
    vr_score: Mapped[float] = mapped_column(Float)
    hr_score: Mapped[float] = mapped_column(Float)
    synergy_score: Mapped[float] = mapped_column(Float)
    ci_lower: Mapped[float] = mapped_column(Float)
    ci_upper: Mapped[float] = mapped_column(Float)
    parameter_version: Mapped[str] = mapped_column(String(20))
    calculation_metadata: Mapped[dict] = mapped_column(JSON, default=dict)
    user: Mapped["User"] = relationship(back_populates="scores")
    """, language="python")

    st.markdown(
        f"**DomainEvent Model**: For reliable event communication (Outbox Pattern).")
    st.code("""
class DomainEvent(Base, TimestampMixin):
    __tablename__ = "domain_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    aggregate_type: Mapped[str] = mapped_column(String(100), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending")
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    __table_args__ = (Index('ix_events_status_created', 'status', 'created_at'),)
    """, language="python")

    st.markdown(f"Alex has now laid out the blueprints for the application's data. The `User` model captures core profile information, `Assessment` tracks user progress, `AIRScore` stores AI-generated evaluation results, and `DomainEvent` is for reliable event communication. The relationships are defined using `relationship` and `ForeignKey`, ensuring data integrity. `cascade=\"all, delete-orphan\"` on `User` relationships simplifies data lifecycle management.")

    if not st.session_state.db_initialized:
        st.info("Please proceed to '2. DB Connectivity & Pooling' to initialize the database and create some sample users before interacting with models.")
    else:
        st.subheader("Interactive Model Demonstration: Create a New User")
        st.markdown(
            f"Demonstrate how to create a new user and observe it being persisted.")
        with st.form("create_new_user_form"):
            new_user_email = st.text_input(
                "New User Email", key="new_user_email_input")
            new_user_name = st.text_input(
                "New User Name", key="new_user_name_input")
            submit_button = st.form_submit_button("Create User")

            if submit_button:
                if new_user_email and new_user_name:
                    try:
                        async def create_new_user_interaction():
                            async for session in get_session():
                                repo = UserRepository(session)
                                new_user_obj = User(email=new_user_email, name=new_user_name,
                                                    occupation_code="TEST", education_level="Bachelors", years_experience=2.0)
                                created_user = await repo.create(new_user_obj)
                                await session.commit()
                                await session.refresh(created_user)
                                st.session_state.created_user_id = created_user.id
                                st.success(
                                    f"User '{created_user.name}' created with ID: {created_user.id}")
                        run_async(create_new_user_interaction())
                    except IntegrityError:
                        st.error(
                            f"User with email '{new_user_email}' already exists. Please use a unique email.")
                    except Exception as e:
                        st.error(f"Error creating user: {e}")
                else:
                    st.warning(
                        "Please provide both email and name for the new user.")

        if st.session_state.created_user_id:
            st.markdown(
                f"**Last Created User ID:** `{st.session_state.created_user_id}`")
            st.markdown(f"You can use this ID in subsequent sections.")

elif st.session_state.current_page == "2. DB Connectivity & Pooling":
    st.title(
        "2. Establishing Asynchronous Database Connectivity and Connection Pooling")

    st.markdown(f"InnovateAI's AI platform needs to handle many concurrent user requests without blocking. Alex knows that synchronous database operations can become a bottleneck, especially with a growing user base. He sets up an asynchronous database connection using SQLAlchemy 2.0 with the `asyncpg` driver and configures connection pooling to efficiently manage database resources. This configuration is essential for maximizing throughput and responsiveness.")

    st.markdown(f"Asynchronous programming is crucial for high-performance I/O-bound applications like web services. Connection pooling prevents the overhead of repeatedly establishing new database connections, improving throughput and responsiveness under load. Without it, each new request might incur the cost of a full database handshake, drastically slowing down the application.")

    st.markdown(r"$$ \Theta = \frac{\text{Number of requests}}{\text{Total time}} $$ Connection Pool Efficiency: The effective connection pool size can be estimated as the number of available connections $$ N_{\text{avail}} $$ out of the maximum pool size $$ N_{\text{max}} $$. A well-tuned pool minimizes latency due to connection acquisition and releases resources promptly. Throughput ($$ \Theta $$), the rate at which requests are processed, is given by the formula above.")
    st.markdown(r"where $$ \Theta $$ is throughput, and asynchronous I/O aims to maximize $$ \Theta $$ by minimizing idle CPU time during I/O wait. Furthermore, transactional guarantees (ACID properties) are crucial. Atomicity ensures that operations within a transaction are all or nothing. Consistency guarantees that a transaction brings the database from one valid state to another. Isolation means concurrent transactions produce the same result as if they were executed sequentially. Durability ensures that once a transaction is committed, it remains committed even in case of power loss.")

    st.subheader("Database Initialization and Sample User Creation")
    if st.button("Initialize In-Memory SQLite Database & Create Sample Users"):
        try:
            user1, user2 = run_async(run_db_setup_and_create_user())
            st.session_state.db_initialized = True
            if user1:
                st.session_state.user_alex_id = user1.id
                st.success(
                    f"Initialized DB and created user: '{user1.name}' (ID: {user1.id})")
            if user2:
                st.session_state.user_jane_id = user2.id
                st.success(f"Created user: '{user2.name}' (ID: {user2.id})")

            st.info(
                "Database schema initialized and sample users created. You can now proceed to other sections.")
        except Exception as e:
            st.error(f"Error initializing database or creating users: {e}")

    if st.session_state.db_initialized:
        st.success("Database is initialized and ready!")
        st.markdown(
            f"**Alex Smith User ID:** `{st.session_state.user_alex_id}`")
        st.markdown(f"**Jane Doe User ID:** `{st.session_state.user_jane_id}`")
        st.markdown(f"The setup of `async_engine` and `AsyncSessionLocal` is central to Alex's async strategy. The `get_session` context manager ensures that database connections are properly acquired and released. The `pool_size` and `max_overflow` parameters are crucial for connection pooling, allowing the application to reuse existing connections and handle spikes in demand gracefully.")
    else:
        st.warning("Database not initialized. Please click the button above.")

elif st.session_state.current_page == "3. Repository Pattern & N+1":
    st.title("3. Implementing the Repository Pattern and Solving N+1 Queries")

    st.markdown(f"To maintain a clean architecture and facilitate easier testing, Alex implements the Repository Pattern, abstracting database operations from the service layer. He also anticipates a common performance pitfall: the N+1 query problem, which arises when fetching a collection of parent objects and then, for each parent, executing a separate query to fetch its child objects. This can drastically degrade performance, especially when dealing with many related records. Alex addresses this with SQLAlchemy's eager loading techniques.")

    st.markdown(f"The Repository Pattern centralizes data access logic, making it easier to manage, test, and potentially swap out ORM or database technologies in the future. The N+1 query problem occurs when loading $N$ parent objects (e.g., users) and then subsequently executing $N$ additional queries to fetch their related child objects (e.g., scores), resulting in $N+1$ queries in total. This is inefficient. Eager loading techniques like `selectinload` reduce this to $1$ or $2$ queries, improving performance significantly.")

    if not st.session_state.db_initialized:
        st.warning(
            "Please initialize the database in the 'DB Connectivity & Pooling' section first.")
    else:
        st.subheader("Create Sample Scores for N+1 Demonstration")
        user_to_add_score_to = st.text_input(
            "User ID to add scores for (e.g., Alex's ID)", value=st.session_state.user_alex_id or "", key="n1_user_id_input")
        if st.button("Add 3 Sample Scores for User"):
            if user_to_add_score_to:
                try:
                    async def add_scores():
                        async for session in get_session():
                            for i in range(3):
                                # create_sample_airscore commits the session, so no need for a session.commit() here
                                await create_sample_airscore(user_to_add_score_to, occupation_code=f"DEV_ENG_{i+1}", session=session)
                                # Ensure distinct timestamps
                                await asyncio.sleep(0.01)
                            st.success(
                                f"3 sample scores added for user ID: {user_to_add_score_to}")
                    run_async(add_scores())
                except Exception as e:
                    st.error(f"Error adding scores: {e}")
            else:
                st.warning("Please enter a User ID.")

        st.subheader("Demonstrating N+1 vs. Eager Loading")
        st.markdown(f"Observe the difference in fetching related data. A 'Simulated N+1' call will trigger multiple database queries (internally by lazy loading) compared to 'Eager Loading' which fetches all related data in fewer queries.")
        user_to_fetch = st.text_input("User ID to fetch (e.g., Alex's ID or other created user)",
                                      value=st.session_state.user_alex_id or "", key="fetch_user_id_n1")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Fetch User & Scores (Simulated N+1)", key="n1_button"):
                if user_to_fetch:
                    start_time = time.time()
                    try:
                        async def fetch_n1():
                            async for session in get_session():
                                repo = UserRepository(session)
                                user = await repo.get_by_id(user_to_fetch)
                                if user:
                                    scores_info = []
                                    # Accessing `user.scores` here, if not eagerly loaded,
                                    # would typically trigger N additional queries (one per score)
                                    # due to SQLAlchemy's default lazy loading.
                                    # For this demo, we illustrate the *concept* of N+1.
                                    for score in user.scores:
                                        scores_info.append(
                                            f"Score ID: {score.id}, AIR Score: {score.air_score}")
                                    st.write(
                                        f"**User (ID: {user.id}, Email: {user.email})**")
                                    st.write(f"**Scores (Simulated N+1):**")
                                    for s_info in scores_info:
                                        st.markdown(f"- {s_info}")
                                else:
                                    st.warning("User not found.")
                        run_async(fetch_n1())
                        end_time = time.time()
                        st.info(
                            f"Time taken (Simulated N+1): {end_time - start_time:.4f} seconds")
                    except Exception as e:
                        st.error(f"Error fetching with N+1: {e}")
                else:
                    st.warning("Please enter a User ID.")

        with col2:
            if st.button("Fetch User & Scores (Eager Loading)", key="eager_button"):
                if user_to_fetch:
                    start_time = time.time()
                    try:
                        async def fetch_eager():
                            async for session in get_session():
                                repo = UserRepository(session)
                                # Uses selectinload
                                user = await repo.get_user_with_scores_eager(user_to_fetch)
                                if user:
                                    st.session_state.retrieved_user_with_scores = user
                                    st.write(
                                        f"**User (ID: {user.id}, Email: {user.email})**")
                                    st.write(f"**Scores (Eager Loaded):**")
                                    if user.scores:
                                        for score in user.scores:
                                            st.markdown(
                                                f"- Score ID: {score.id}, AIR Score: {score.air_score}")
                                    else:
                                        st.markdown("- No scores found.")
                                else:
                                    st.warning("User not found.")
                        run_async(fetch_eager())
                        end_time = time.time()
                        st.info(
                            f"Time taken (Eager Loading): {end_time - start_time:.4f} seconds")
                    except Exception as e:
                        st.error(f"Error fetching with eager loading: {e}")
                else:
                    st.warning("Please enter a User ID.")

        st.markdown(f"By implementing `UserRepository`, Alex has created a clean boundary between the business logic and data access. The `get_user_with_scores_eager` method, utilizing `selectinload(User.scores)`, directly addresses the N+1 query problem by fetching related `AIRScore` objects in a minimal number of queries. This significantly reduces database load and improves response times, as observed by the difference in execution times.")

elif st.session_state.current_page == "4. Caching with Redis":
    st.title("4. Optimizing Data Access with Redis Caching Strategies")

    st.markdown(f"InnovateAI's user profiles and their latest AI scores are frequently accessed, especially during the initial loading of the user dashboard. To offload the primary PostgreSQL database and accelerate response times for these hot data points, Alex decides to implement a caching layer using Redis. He needs to consider a read-through caching strategy, where data is fetched from the cache if available, otherwise from the database and then stored in the cache for subsequent requests.")

    st.markdown(f"Caching is critical for high-performance applications, reducing latency and database load by storing frequently accessed data in a fast, in-memory store like Redis. The read-through strategy is robust for frequently read, less frequently updated data. It simplifies cache management by encapsulating the cache-or-DB logic.")

    st.markdown(r"$$ H = \frac{\text{Number of Cache Hits}}{\text{Total Number of Requests}} $$ The effectiveness of caching is measured by the Cache Hit Ratio ($$ H $$). where $$ H $$ is the cache hit ratio. A higher $$ H $$ indicates better cache effectiveness.")
    st.markdown(r"$$ T_{\text{avg}} = H \times T_{\text{cache}} + (1-H) \times (T_{\text{cache}} + T_{\text{database}}) $$ The Average Access Time ($$ T_{\text{avg}} $$) with caching is given by the formula above, where $$ T_{\text{cache}} $$ is cache access time and $$ T_{\text{database}} $$ is database access time. A good caching strategy aims to minimize $$ T_{\text{avg}} $$.")

    if not st.session_state.db_initialized:
        st.warning(
            "Please initialize the database in the 'DB Connectivity & Pooling' section first.")
    else:
        st.subheader("Interactive Caching Demonstration")

        # Ensure a user exists for caching demo
        if not st.session_state.user_for_caching_id:
            try:
                async def create_cache_user_and_score():
                    async for session in get_session():
                        # Check if a specific user (e.g., Jane) already exists for demo continuity
                        user = await UserRepository(session).get_by_email("jane.doe@innovateai.com")
                        if not user:  # Fallback if Jane isn't there
                            new_user = User(
                                email="cache_demo@innovateai.com", name="Cache Demo User")
                            user = await UserRepository(session).create(new_user)
                            await session.commit()
                            await session.refresh(user)
                        st.session_state.user_for_caching_id = user.id

                        # Create a sample score if none exists for this user
                        latest_score = await UserRepository(session).get_latest_score(user.id)
                        if not latest_score:
                            score = await create_sample_airscore(user.id, occupation_code="CACHE_EX", session=session)
                            st.session_state.latest_airscore_id = score.id
                        else:
                            st.session_state.latest_airscore_id = latest_score.id

                        st.info(
                            f"User for caching demo (ID: `{st.session_state.user_for_caching_id}`) and latest score (ID: `{st.session_state.latest_airscore_id}`) ready.")
                run_async(create_cache_user_and_score())
            except Exception as e:
                st.error(f"Error preparing user for caching demo: {e}")

        user_id_for_cache = st.session_state.user_for_caching_id

        if user_id_for_cache:
            st.markdown(
                f"Using **User ID**: `{user_id_for_cache}` for caching demonstration.")
            st.markdown(
                f"**Latest AIRScore ID**: `{st.session_state.latest_airscore_id}`")

            col1_cache, col2_cache = st.columns(2)
            with col1_cache:
                if st.button("Fetch User (Cached)", key="fetch_user_cached_btn"):
                    start_time = time.time()
                    try:
                        async def fetch_user_cached_interaction():
                            async for session in get_session():
                                cached_repo = CachedUserRepository(
                                    session, REDIS_CLIENT)
                                user = await cached_repo.get_by_id_cached(user_id_for_cache)
                                if user:
                                    st.write(f"**Cached User Details:**")
                                    st.json({
                                        "id": user.id, "email": user.email, "name": user.name,
                                        "occupation_code": user.occupation_code,
                                        "created_at": user.created_at.isoformat() if user.created_at else None
                                    })
                                else:
                                    st.warning(
                                        "User not found in cache or DB.")
                        run_async(fetch_user_cached_interaction())
                        end_time = time.time()
                        st.markdown(
                            f"Time taken: {end_time - start_time:.4f} seconds")
                        # Simplified cache hit/miss tracking based on presence of REDIS_CLIENT.get call
                        # For mock, REDIS_CLIENT.get.called will be True if called. return_value is None on mock miss.
                        # Check if the cache was actually queried
                        if hasattr(REDIS_CLIENT.get, 'called') and REDIS_CLIENT.get.called:
                            # Check if the mock client returned a non-None value to simulate a hit
                            # For a real client, data will be non-None on hit.
                            if REDIS_CLIENT.get.return_value is not None:
                                st.session_state.cache_hits += 1
                            else:
                                st.session_state.cache_misses += 1
                            REDIS_CLIENT.get.reset_mock()  # Reset mock for next call if using mock client
                    except Exception as e:
                        st.error(f"Error fetching user with cache: {e}")

            with col2_cache:
                if st.button("Fetch Latest AIRScore (Cached)", key="fetch_score_cached_btn"):
                    start_time = time.time()
                    try:
                        async def fetch_score_cached_interaction():
                            async for session in get_session():
                                cached_repo = CachedUserRepository(
                                    session, REDIS_CLIENT)
                                score = await cached_repo.get_latest_score_cached(user_id_for_cache)
                                if score:
                                    st.write(
                                        f"**Cached Latest AIRScore Details:**")
                                    st.json({
                                        "id": score.id, "user_id": score.user_id,
                                        "air_score": score.air_score, "occupation": score.occupation_code,
                                        "parameter_version": score.parameter_version,
                                        "created_at": score.created_at.isoformat() if score.created_at else None
                                    })
                                else:
                                    st.warning(
                                        "Latest AIRScore not found in cache or DB.")
                        run_async(fetch_score_cached_interaction())
                        end_time = time.time()
                        st.markdown(
                            f"Time taken: {end_time - start_time:.4f} seconds")
                        if hasattr(REDIS_CLIENT.get, 'called') and REDIS_CLIENT.get.called:
                            if REDIS_CLIENT.get.return_value is not None:
                                st.session_state.cache_hits += 1
                            else:
                                st.session_state.cache_misses += 1
                            REDIS_CLIENT.get.reset_mock()
                    except Exception as e:
                        st.error(f"Error fetching score with cache: {e}")

            st.markdown(f"---")
            st.subheader("Cache Invalidation")
            st.markdown(
                f"After an update, old cached data needs to be purged. Invalidate the cache for User ID: `{user_id_for_cache}`.")
            if st.button("Invalidate Cache for this User", key="invalidate_cache_btn"):
                try:
                    async def invalidate_cache_interaction():
                        # The session might not be strictly needed for cache invalidation for the redis client,
                        # but we pass it for consistency with repository pattern.
                        async for session in get_session():
                            cached_repo = CachedUserRepository(
                                session, REDIS_CLIENT)
                            await cached_repo.invalidate_user_cache(user_id_for_cache)
                            st.success(
                                f"Cache invalidated for user ID: {user_id_for_cache}")
                    run_async(invalidate_cache_interaction())
                except Exception as e:
                    st.error(f"Error invalidating cache: {e}")

            st.markdown(f"---")
            st.subheader("Cache Metrics (Simulated)")
            total_requests = st.session_state.cache_hits + st.session_state.cache_misses
            hit_ratio = (st.session_state.cache_hits /
                         total_requests) if total_requests > 0 else 0
            st.markdown(f"**Cache Hits:** {st.session_state.cache_hits}")
            st.markdown(f"**Cache Misses:** {st.session_state.cache_misses}")
            st.markdown(f"**Total Requests:** {total_requests}")
            st.markdown(f"**Cache Hit Ratio:** {hit_ratio:.2f}")

            st.markdown(f"Alex's implementation of `CachedUserRepository` successfully demonstrates a read-through caching strategy. The first request will typically result in a 'Cache MISS' (fetching from DB), and subsequent requests for the same data show a 'Cache HIT' (retrieving directly from Redis). The `invalidate_user_cache` method ensures data consistency by clearing stale cache entries after updates. This optimizes data access for read-heavy operations, improving the user experience.")

elif st.session_state.current_page == "5. Eventing (Outbox Pattern)":
    st.title("5. Building a Reliable Eventing System with the Outbox Pattern")

    st.markdown(f"Alex is tasked with ensuring that critical domain events (like an `AIRScore` being calculated or an `Assessment` completing) are reliably published to other microservices within InnovateAI, even in the face of temporary network issues or consumer downtime. He implements the Outbox Pattern, using the `DomainEvent` table as a robust buffer. This pattern is vital for ensuring that services remain loosely coupled and that event-driven architectures maintain data consistency.")

    st.markdown(f"In distributed systems, reliable communication between services is paramount. The Outbox Pattern ensures atomicity: a business operation and the recording of its corresponding domain event happen within a single database transaction. This guarantees that events are never lost if the publishing mechanism fails after the business operation succeeds but before the event is sent to the message broker. It leverages the ACID properties of the database.")

    st.markdown(f"The Outbox Pattern helps achieve **eventual consistency**. In a distributed system, data might not be immediately consistent across all services, but it will eventually converge. By guaranteeing events are eventually delivered, the Outbox pattern facilitates this convergence without requiring a complex two-phase commit protocol across services.")

    if not st.session_state.db_initialized:
        st.warning(
            "Please initialize the database in the 'DB Connectivity & Pooling' section first.")
    else:
        st.subheader("Generate an AIRScore and a Pending Domain Event")

        # Ensure a user exists for eventing demo
        if not st.session_state.event_user_id:
            try:
                async def create_event_user():
                    async for session in get_session():
                        user = await UserRepository(session).get_by_email("event_demo@innovateai.com")
                        if not user:
                            new_user = User(
                                email="event_demo@innovateai.com", name="Event Demo User")
                            user = await UserRepository(session).create(new_user)
                            await session.commit()
                            await session.refresh(user)
                        st.session_state.event_user_id = user.id
                        st.info(
                            f"User for eventing demo (ID: `{st.session_state.event_user_id}`) ready.")
                run_async(create_event_user())
            except Exception as e:
                st.error(f"Error preparing user for eventing demo: {e}")

        user_id_for_event = st.session_state.event_user_id
        if user_id_for_event:
            st.markdown(
                f"Using **User ID**: `{user_id_for_event}` for eventing demonstration.")

            if st.button("Calculate & Store AIRScore (Creates Pending Event)", key="create_event_btn"):
                try:
                    async def calculate_score_and_event_interaction():
                        st.info("Calling `calculate_and_store_airscore`...")
                        score, event = await calculate_and_store_airscore(user_id_for_event, "AI_ASSESSOR")
                        st.success(
                            f"AIRScore (ID: `{score.id}`) calculated and DomainEvent (ID: `{event.id}`, Type: `{event.event_type}`) recorded as 'pending'.")

                    run_async(calculate_score_and_event_interaction())
                except Exception as e:
                    st.error(
                        f"Error calculating score and recording event: {e}")

            st.subheader("Event Publisher Status")
            col1_pub, col2_pub = st.columns(2)
            with col1_pub:
                if st.button("Start Event Publisher (Background)", key="start_publisher_btn", disabled=st.session_state.event_publisher_running):
                    st.session_state.event_publisher_running = True
                    st.session_state.publisher_stop_event.clear()

                    async def run_publisher_in_background():
                        st.info(
                            "Event publisher starting... Processing up to 3 cycles of events.")
                        processed_count = 0
                        for _ in range(3):  # Simulate 3 polling cycles
                            if st.session_state.publisher_stop_event.is_set():
                                break
                            async for session in get_session():  # Acquire a session for this cycle
                                event_repo = DomainEventRepository(session)
                                pending_events = await event_repo.get_pending_events(limit=5)
                                if pending_events:
                                    event_ids_to_publish = [
                                        event.id for event in pending_events]
                                    # This method commits
                                    await event_repo.mark_as_published(event_ids_to_publish)
                                    processed_count += len(event_ids_to_publish)
                                    st.markdown(
                                        f"*(Publisher activity)* Processed {len(event_ids_to_publish)} events.")
                                await asyncio.sleep(0.5)
                        st.session_state.event_publisher_running = False
                        st.success(
                            f"Event publisher finished its simulated run, processed {processed_count} events.")

                    # Use a new event loop or ensure this runs in the existing one without blocking Streamlit's main loop
                    # For simplicity in Streamlit, directly calling asyncio.run in a button callback
                    # means it will block until done. If a truly background process is needed
                    # in a real app, it would be a separate thread/process.
                    run_async(run_publisher_in_background())
                    st.rerun()

            with col2_pub:
                if st.button("Stop Event Publisher", key="stop_publisher_btn", disabled=not st.session_state.event_publisher_running):
                    st.session_state.publisher_stop_event.set()
                    st.session_state.event_publisher_running = False
                    st.warning(
                        "Event publisher signalled to stop (will stop after current cycle).")
                    st.rerun()

            st.subheader("Current Event Status")

            async def refresh_events_status_internal():
                async for session in get_session():
                    event_repo = DomainEventRepository(session)
                    pending_events = await event_repo.get_pending_events(limit=10)
                    all_events_result = await session.execute(select(DomainEvent).order_by(DomainEvent.created_at.desc()))
                    all_events = all_events_result.scalars().all()

                    st.session_state.pending_events_display = [
                        {"ID": e.id, "Type": e.event_type, "Status": e.status,
                            "Created": e.created_at.strftime("%Y-%m-%d %H:%M:%S")}
                        for e in pending_events
                    ]
                    st.session_state.processed_events_display = [
                        {"ID": e.id, "Type": e.event_type, "Status": e.status, "Published": e.published_at.strftime(
                            "%Y-%m-%d %H:%M:%S") if e.published_at else "N/A"}
                        for e in all_events if e.status == "published"
                    ]

            if st.button("Refresh Event Status Now", key="refresh_event_status_btn"):
                run_async(refresh_events_status_internal())

            st.markdown("---")
            st.markdown(f"**Pending Events (Awaiting Publication)**")
            if st.session_state.pending_events_display:
                st.table(st.session_state.pending_events_display)
            else:
                st.info("No pending events.")

            st.markdown(f"**Published Events (Processed)**")
            if st.session_state.processed_events_display:
                st.table(st.session_state.processed_events_display)
            else:
                st.info("No published events yet.")

            st.markdown(f"Alex's implementation clearly demonstrates the Outbox Pattern. When `calculate_and_store_airscore` is called, both the `AIRScore` record and the `DomainEvent` are created within a single database transaction, guaranteeing atomicity. The `DomainEvent` is initially marked `pending`.")
            st.markdown(f"The simulated `event_publisher` then periodically polls the `domain_events` table for `pending` events. Upon finding them, it simulates publishing them and then updates their status to `published` in the database. This two-phase commit strategy ensures no event is ever lost, even if an external message broker is temporarily unavailable. Alex can verify this process by observing the status transitions.")


# License
st.caption('''
---
## QuantUniversity License

© QuantUniversity 2025  
This notebook was created for **educational purposes only** and is **not intended for commercial use**.  

- You **may not copy, share, or redistribute** this notebook **without explicit permission** from QuantUniversity.  
- You **may not delete or modify this license cell** without authorization.  
- This notebook was generated using **QuCreate**, an AI-powered assistant.  
- Content generated by AI may contain **hallucinated or incorrect information**. Please **verify before using**.  

All rights reserved. For permissions or commercial licensing, contact: [info@qusandbox.com](mailto:info@qusandbox.com)
''')
