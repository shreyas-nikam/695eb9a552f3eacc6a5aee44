from datetime import datetime
from enum import Enum
import uuid
from typing import Optional, List, Dict, Any
import asyncio
import json

from sqlalchemy import String, DateTime, Float, ForeignKey, JSON, func, Index, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base, selectinload
from sqlalchemy.exc import IntegrityError

import redis.asyncio as aioredis
import time
import asyncio
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base # Added for Base and model definitions
from sqlalchemy import String, DateTime, Float, ForeignKey, JSON, func, Index, select # Added for model definitions
from sqlalchemy.exc import IntegrityError # Added for exception handling
from datetime import datetime # Added for TimestampMixin
import uuid # Added for UUID generation
from typing import Optional, List # Added for type hints


# Base class for declarative models (Copied from cell 619b339f)
Base = declarative_base()

# Mixin for common timestamp fields (Copied from cell 619b339f)
class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

# User model (Copied from cell 619b339f)
class User(Base, TimestampMixin):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    occupation_code: Mapped[Optional[str]] = mapped_column(String(20))
    education_level: Mapped[Optional[str]] = mapped_column(String(50))
    years_experience: Mapped[Optional[float]] = mapped_column(Float)

    # Relationships
    assessments: Mapped[List["Assessment"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    scores: Mapped[List["AIRScore"]] = relationship(back_populates="user", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id='{self.id}', email='{self.email}')>"

# Assessment model (Copied from cell 619b339f)
class Assessment(Base, TimestampMixin):
    __tablename__ = "assessments"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    status: Mapped[str] = mapped_column(String(20), default="in_progress") # e.g., in_progress, completed, cancelled
    component: Mapped[str] = mapped_column(String(50)) # e.g., fluency, domain, adaptive
    current_ability: Mapped[float] = mapped_column(Float, default=0.0)
    items_administered: Mapped[int] = mapped_column(default=0)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="assessments")

    def __repr__(self):
        return f"<Assessment(id='{self.id}', user_id='{self.user_id}', status='{self.status}')>"

# AIRScore model (Copied from cell 619b339f)
class AIRScore(Base, TimestampMixin):
    __tablename__ = "air_scores"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id"), index=True)
    occupation_code: Mapped[str] = mapped_column(String(20))

    # Scores
    air_score: Mapped[float] = mapped_column(Float)
    vr_score: Mapped[float] = mapped_column(Float)
    hr_score: Mapped[float] = mapped_column(Float)
    synergy_score: Mapped[float] = mapped_column(Float)

    # Confidence interval
    ci_lower: Mapped[float] = mapped_column(Float)
    ci_upper: Mapped[float] = mapped_column(Float)

    # Metadata
    parameter_version: Mapped[str] = mapped_column(String(20))
    calculation_metadata: Mapped[dict] = mapped_column(JSON, default=dict)

    # Relationships
    user: Mapped["User"] = relationship(back_populates="scores")

    def __repr__(self):
        return f"<AIRScore(id='{self.id}', user_id='{self.user_id}', air_score={self.air_score})>"


# Database URL for PostgreSQL with asyncpg driver
# Changed to in-memory SQLite for self-contained notebook execution without requiring a running PostgreSQL instance.
DATABASE_URL = "sqlite+aiosqlite:///:memory:"

# Create the asynchronous engine
# pool_size and max_overflow help configure the connection pool
async_engine = create_async_engine(DATABASE_URL, echo=False)

# Create an async session maker
AsyncSessionLocal = async_sessionmaker(autocommit=False, autoflush=False, bind=async_engine, expire_on_commit=False)

# Async context manager for database sessions
async def get_session() -> AsyncSession:
    async with AsyncSessionLocal() as session:
        yield session

# Function to initialize the database schema (similar to Alembic upgrade head)
async def initialize_db():
    print("Initializing database schema...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("Database schema initialized.")

# Demonstrate creating a user
async def create_sample_user(email: str, name: str) -> User:
    async for session in get_session():
        try:
            new_user = User(email=email, name=name, occupation_code="SW_ENG", education_level="Masters", years_experience=5.0)
            session.add(new_user)
            await session.commit()
            # No need to refresh immediately after commit if expire_on_commit=False and object is new
            # await session.refresh(new_user)
            print(f"Created user: {new_user}")
            return new_user
        except IntegrityError:
            await session.rollback()
            print(f"User with email {email} already exists. Rolling back.")
            # Fetch existing user if needed, but for demo, just return None
            return None
        except Exception as e:
            await session.rollback()
            print(f"Error creating user: {e}")
            raise

# Initialize DB and create a user
async def run_db_setup_and_create_user():
    await initialize_db()
    user1 = await create_sample_user("alex.smith@innovateai.com", "Alex Smith")
    user2 = await create_sample_user("jane.doe@innovateai.com", "Jane Doe")
    return user1, user2

user_alex, user_jane = await run_db_setup_and_create_user()
import pytest
import asyncio
from datetime import datetime, timedelta
import uuid
from typing import Optional, List, Dict, Any
from enum import Enum
import time

from sqlalchemy import String, DateTime, Float, ForeignKey, JSON, func, Index, select
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship, declarative_base, selectinload
from sqlalchemy.exc import IntegrityError
from sqlalchemy.engine.reflection import Inspector # For inspecting tables


# Re-define models here for a self-contained test file
Base = declarative_base()


# Fixture for an async SQLite session for tests
@pytest.fixture(scope="module")
def event_loop():
    """Provides the event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def async_session_fixture():
    """
    Provides an in-memory SQLite database session for each test function.
    Tables are created and dropped for each test.
    """
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with AsyncSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


# --- Code to test (UserRepository and create_sample_airscore) ---

class UserRepository:
    """Repository for user data access."""
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_id(self, user_id: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_by_email(self, email: str) -> Optional[User]:
        result = await self.session.execute(
            select(User).where(User.email == email)
        )
        return result.scalar_one_or_none()

    async def create(self, user: User) -> User:
        self.session.add(user)
        await self.session.flush() # flush to get ID, but don't commit yet
        return user

    async def get_latest_score(self, user_id: str) -> Optional[AIRScore]:
        result = await self.session.execute(
            select(AIRScore)
            .where(AIRScore.user_id == user_id)
            .order_by(AIRScore.created_at.desc())
            .limit(1)
        )
        return result.scalar_one_or_none()

    async def get_user_with_scores_eager(self, user_id: str) -> Optional[User]:
        """Fetches a user and eagerly loads their scores to avoid N+1 problem."""
        result = await self.session.execute(
            select(User)
            .options(selectinload(User.scores)) # Eagerly load scores
            .where(User.id == user_id)
        )
        return result.scalar_one_or_none()

    async def get_all_users_with_scores_eager(self) -> List[User]:
        """Fetches all users and eagerly loads their scores."""
        result = await self.session.execute(
            select(User).options(selectinload(User.scores))
        )
        return result.scalars().all()


async def create_sample_airscore(user_id: str, occupation_code: str = "SW_ENG", session: AsyncSession = None) -> AIRScore:
    """
    Creates a sample AIRScore. Modified to accept a session directly for testing.
    If no session is provided, it attempts to use a mockable get_session (for patching).
    """
    if session is None:
        raise ValueError("Session must be provided for create_sample_airscore in tests.")

    try:
        new_score = AIRScore(
            user_id=user_id,
            occupation_code=occupation_code,
            air_score=85.5, vr_score=0.7, hr_score=0.8, synergy_score=0.9,
            ci_lower=0.8, ci_upper=0.9,
            parameter_version="v1.2",
            calculation_metadata={"model": "deep_eval_v3", "factors": ["fluency", "domain"]}
        )
        session.add(new_score)
        await session.commit()
        await session.refresh(new_score)
        return new_score
    except Exception:
        await session.rollback()
        raise


# --- Pytest Unit Tests ---

@pytest.mark.asyncio
async def test_user_repository_init(async_session_fixture: AsyncSession):
    """Tests that the UserRepository initializes with the given session."""
    repo = UserRepository(async_session_fixture)
    assert repo.session is async_session_fixture


@pytest.mark.asyncio
async def test_user_repository_create(async_session_fixture: AsyncSession):
    """Tests the create method of UserRepository."""
    repo = UserRepository(async_session_fixture)
    new_user = User(email="create_test@example.com", name="Create Test User")

    created_user = await repo.create(new_user) # create flushes, not commits

    assert created_user is new_user # Ensure the same object is returned
    assert created_user.id is not None # ID should be assigned after flush
    assert created_user.email == "create_test@example.com"

    # Commit the session to make the user persistent for subsequent verification
    await async_session_fixture.commit()

    # Verify user is truly in the database using a separate query
    retrieved_user_query = await async_session_fixture.execute(
        select(User).where(User.id == created_user.id)
    )
    retrieved_user = retrieved_user_query.scalar_one_or_none()
    assert retrieved_user is not None
    assert retrieved_user.email == created_user.email


@pytest.mark.asyncio
async def test_user_repository_get_by_id(async_session_fixture: AsyncSession):
    """Tests retrieving a user by ID."""
    user = User(email="id_test@example.com", name="ID Test User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    repo = UserRepository(async_session_fixture)
    found_user = await repo.get_by_id(user.id)
    assert found_user is not None
    assert found_user.id == user.id
    assert found_user.email == user.email

    not_found_user = await repo.get_by_id(str(uuid.uuid4()))
    assert not_found_user is None


@pytest.mark.asyncio
async def test_user_repository_get_by_email(async_session_fixture: AsyncSession):
    """Tests retrieving a user by email."""
    user = User(email="email_test@example.com", name="Email Test User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    repo = UserRepository(async_session_fixture)
    found_user = await repo.get_by_email(user.email)
    assert found_user is not None
    assert found_user.id == user.id
    assert found_user.email == user.email

    not_found_user = await repo.get_by_email("nonexistent@example.com")
    assert not_found_user is None


@pytest.mark.asyncio
async def test_create_sample_airscore_success(async_session_fixture: AsyncSession):
    """Tests successful creation of an AIRScore using the helper function."""
    user = User(email="score_user@example.com", name="Score User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    score = await create_sample_airscore(user.id, session=async_session_fixture)

    assert score is not None
    assert score.user_id == user.id
    assert score.occupation_code == "SW_ENG"
    assert score.air_score == 85.5
    assert score.parameter_version == "v1.2"
    assert score.calculation_metadata == {"model": "deep_eval_v3", "factors": ["fluency", "domain"]}

    # Verify it's in the DB
    retrieved_score_query = await async_session_fixture.execute(
        select(AIRScore).where(AIRScore.id == score.id)
    )
    retrieved_score = retrieved_score_query.scalar_one_or_none()
    assert retrieved_score is not None
    assert retrieved_score.air_score == score.air_score


@pytest.mark.asyncio
async def test_user_repository_get_latest_score(async_session_fixture: AsyncSession):
    """Tests retrieving the latest AIRScore for a user."""
    user = User(email="latest_score@example.com", name="Latest Score User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    repo = UserRepository(async_session_fixture)

    # Case 1: No scores yet
    latest_score = await repo.get_latest_score(user.id)
    assert latest_score is None

    # Case 2: Create first score
    score1 = await create_sample_airscore(user.id, occupation_code="SE-001", session=async_session_fixture)
    await asyncio.sleep(0.01) # Ensure different created_at timestamp

    # Case 3: Create second, later score
    score2 = AIRScore(user_id=user.id, occupation_code="SE-002", air_score=99.9, vr_score=0.9, hr_score=0.9, synergy_score=0.9, ci_lower=0.9, ci_upper=0.9, parameter_version="v1.3")
    async_session_fixture.add(score2)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(score2)

    latest_score = await repo.get_latest_score(user.id)
    assert latest_score is not None
    assert latest_score.id == score2.id # score2 should be the latest due to higher timestamp

    # Test for non-existent user
    non_existent_score = await repo.get_latest_score(str(uuid.uuid4()))
    assert non_existent_score is None


@pytest.mark.asyncio
async def test_user_repository_get_user_with_scores_eager(async_session_fixture: AsyncSession):
    """
    Tests eager loading of user scores and verifies the relationship loading.
    """
    user = User(email="eager_load_user@example.com", name="Eager Load User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    score1 = await create_sample_airscore(user.id, occupation_code="EAGER1", session=async_session_fixture)
    score2 = await create_sample_airscore(user.id, occupation_code="EAGER2", session=async_session_fixture)
    await async_session_fixture.commit() # Commit scores for the user

    repo = UserRepository(async_session_fixture)

    # Get user with eager loading
    loaded_user = await repo.get_user_with_scores_eager(user.id)

    assert loaded_user is not None
    assert loaded_user.id == user.id
    assert len(loaded_user.scores) == 2

    # Check if the relationship was loaded eagerly using internal ORM state
    # For selectinload, the collection will be directly present and populated.
    assert hasattr(loaded_user, 'scores')
    assert loaded_user.scores[0].user is loaded_user or loaded_user.scores[1].user is loaded_user

    score_ids = {s.id for s in loaded_user.scores}
    assert score1.id in score_ids
    assert score2.id in score_ids

    # Test non-existent user
    non_existent_user = await repo.get_user_with_scores_eager(str(uuid.uuid4()))
    assert non_existent_user is None


@pytest.mark.asyncio
async def test_user_repository_get_all_users_with_scores_eager(async_session_fixture: AsyncSession):
    """
    Tests eager loading of scores for all users.
    """
    user1 = User(email="all_user1@example.com", name="All User 1")
    user2 = User(email="all_user2@example.com", name="All User 2")
    async_session_fixture.add_all([user1, user2])
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user1)
    await async_session_fixture.refresh(user2)

    score1_u1 = await create_sample_airscore(user1.id, occupation_code="U1S1", session=async_session_fixture)
    score2_u1 = await create_sample_airscore(user1.id, occupation_code="U1S2", session=async_session_fixture)
    score1_u2 = await create_sample_airscore(user2.id, occupation_code="U2S1", session=async_session_fixture)
    await async_session_fixture.commit() # Commit all scores

    repo = UserRepository(async_session_fixture)
    all_users = await repo.get_all_users_with_scores_eager()

    assert len(all_users) == 2

    user1_found = next((u for u in all_users if u.id == user1.id), None)
    user2_found = next((u for u in all_users if u.id == user2.id), None)

    assert user1_found is not None
    assert user2_found is not None

    assert len(user1_found.scores) == 2
    assert {s.id for s in user1_found.scores} == {score1_u1.id, score2_u1.id}
    assert hasattr(user1_found, 'scores') # Check if relationship attribute exists and is loaded

    assert len(user2_found.scores) == 1
    assert {s.id for s in user2_found.scores} == {score1_u2.id}
    assert hasattr(user2_found, 'scores') # Check if relationship attribute exists and is loaded

    # Test with no users in an empty database
    # Need to clean up the current session and re-initialize for this specific sub-test
    await async_session_fixture.rollback()
    await async_session_fixture.execute(User.__table__.delete())
    await async_session_fixture.execute(AIRScore.__table__.delete())
    await async_session_fixture.commit()
    async_session_fixture.expire_all() # Ensure objects are not in session state

    no_users_repo = UserRepository(async_session_fixture)
    no_users = await no_users_repo.get_all_users_with_scores_eager()
    assert len(no_users) == 0
# --- Fixtures ---

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def async_session_fixture():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with AsyncSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()

@pytest.fixture
def mock_redis_client(mocker):
    """Mocks the aioredis.Redis client."""
    mock_client = mocker.AsyncMock(spec=aioredis.Redis)
    mock_client.get.return_value = None # Default: cache miss
    return mock_client

@pytest.fixture
def cached_user_repo_fixture(async_session_fixture: AsyncSession, mock_redis_client: mock.AsyncMock):
    """Provides an instance of CachedUserRepository for tests."""
    return CachedUserRepository(async_session_fixture, mock_redis_client)


# --- Code to be tested (CachedUserRepository) ---

# The original Redis client setup is commented out as it's mocked for tests.
# REDIS_URL = "redis://localhost:6379/0"
# redis_client = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

class CachedUserRepository(UserRepository):
    """Repository for user data access with Redis caching."""
    def __init__(self, session: AsyncSession, redis_client: aioredis.Redis):
        super().__init__(session)
        self.redis = redis_client

    def _user_cache_key(self, user_id: str) -> str:
        return f"user:{user_id}"

    def _latest_airscore_cache_key(self, user_id: str) -> str:
        return f"latest_airscore:{user_id}"

    async def get_by_id_cached(self, user_id: str) -> Optional[User]:
        cache_key = self._user_cache_key(user_id)
        cached_user_data = await self.redis.get(cache_key)

        if cached_user_data:
            # print(f"Cache HIT for user {user_id}") # Removed print for tests
            # Deserialize user from JSON string
            user_dict = json.loads(cached_user_data)
            # Create a User object (without deep relationships for simplicity here)
            return User(id=user_dict['id'], email=user_dict['email'], name=user_dict['name'],
                        occupation_code=user_dict.get('occupation_code'),
                        education_level=user_dict.get('education_level'),
                        years_experience=user_dict.get('years_experience'),
                        created_at=datetime.fromisoformat(user_dict['created_at']) if 'created_at' in user_dict else datetime.now(),
                        updated_at=datetime.fromisoformat(user_dict['updated_at']) if 'updated_at' in user_dict else datetime.now()
                        )

        # print(f"Cache MISS for user {user_id}. Fetching from DB.") # Removed print for tests
        user = await super().get_by_id(user_id)
        if user:
            # Serialize user to JSON and store in cache
            user_dict = {
                "id": user.id, "email": user.email, "name": user.name,
                "occupation_code": user.occupation_code, "education_level": user.education_level,
                "years_experience": user.years_experience,
                "created_at": user.created_at.isoformat(),
                "updated_at": user.updated_at.isoformat()
            }
            await self.redis.set(cache_key, json.dumps(user_dict), ex=3600) # Cache for 1 hour
        return user

    async def get_latest_score_cached(self, user_id: str) -> Optional[AIRScore]:
        cache_key = self._latest_airscore_cache_key(user_id)
        cached_score_data = await self.redis.get(cache_key)

        if cached_score_data:
            # print(f"Cache HIT for latest AIRScore of user {user_id}") # Removed print for tests
            score_dict = json.loads(cached_score_data)
            # Reconstruct AIRScore object for demo, real scenarios might use Pydantic models
            return AIRScore(
                id=score_dict['id'], user_id=score_dict['user_id'], occupation_code=score_dict['occupation_code'],
                air_score=score_dict['air_score'], vr_score=score_dict['vr_score'], hr_score=score_dict['hr_score'], synergy_score=score_dict['synergy_score'],
                ci_lower=score_dict['ci_lower'], ci_upper=score_dict['ci_upper'],
                parameter_version=score_dict['parameter_version'],
                calculation_metadata=score_dict['calculation_metadata'],
                created_at=datetime.fromisoformat(score_dict['created_at']),
                updated_at=datetime.fromisoformat(score_dict['updated_at'])
            )

        # print(f"Cache MISS for latest AIRScore of user {user_id}. Fetching from DB.") # Removed print for tests
        score = await super().get_latest_score(user_id)
        if score:
            score_dict = {
                "id": score.id, "user_id": score.user_id, "occupation_code": score.occupation_code,
                "air_score": score.air_score, "vr_score": score.vr_score, "hr_score": score.hr_score, "synergy_score": score.synergy_score,
                "ci_lower": score.ci_lower, "ci_upper": score.ci_upper,
                "parameter_version": score.parameter_version,
                "calculation_metadata": score.calculation_metadata,
                "created_at": score.created_at.isoformat(),
                "updated_at": score.updated_at.isoformat()
            }
            await self.redis.set(cache_key, json.dumps(score_dict), ex=300) # Cache for 5 minutes
        return score

    async def invalidate_user_cache(self, user_id: str):
        """Invalidate user cache upon update/delete."""
        await self.redis.delete(self._user_cache_key(user_id))
        await self.redis.delete(self._latest_airscore_cache_key(user_id))
        # print(f"Invalidated cache for user {user_id}") # Removed print for tests


# --- Pytest Unit Tests for CachedUserRepository ---

@pytest.mark.asyncio
async def test_cached_user_repository_init(cached_user_repo_fixture: CachedUserRepository, async_session_fixture: AsyncSession, mock_redis_client: mock.AsyncMock):
    """Tests that CachedUserRepository initializes correctly."""
    assert cached_user_repo_fixture.session is async_session_fixture
    assert cached_user_repo_fixture.redis is mock_redis_client


@pytest.mark.asyncio
async def test_user_cache_key_generation(cached_user_repo_fixture: CachedUserRepository):
    """Tests the user cache key generation method."""
    user_id = str(uuid.uuid4())
    assert cached_user_repo_fixture._user_cache_key(user_id) == f"user:{user_id}"


@pytest.mark.asyncio
async def test_latest_airscore_cache_key_generation(cached_user_repo_fixture: CachedUserRepository):
    """Tests the latest AIRScore cache key generation method."""
    user_id = str(uuid.uuid4())
    assert cached_user_repo_fixture._latest_airscore_cache_key(user_id) == f"latest_airscore:{user_id}"


@pytest.mark.asyncio
async def test_get_by_id_cached_miss_then_db_hit(cached_user_repo_fixture: CachedUserRepository, async_session_fixture: AsyncSession, mock_redis_client: mock.AsyncMock):
    """Tests get_by_id_cached for a cache miss scenario where user is found in DB."""
    # Setup: User in DB, not in cache
    user = User(email="cached_miss@example.com", name="Cached Miss User",
                occupation_code="SW_ENG", education_level="Masters", years_experience=5.0)
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    mock_redis_client.get.return_value = None # Ensure cache miss

    # Action
    retrieved_user = await cached_user_repo_fixture.get_by_id_cached(user.id)

    # Assertions
    assert retrieved_user is not None
    assert retrieved_user.id == user.id
    assert retrieved_user.email == user.email
    assert retrieved_user.name == user.name
    assert retrieved_user.occupation_code == user.occupation_code
    assert retrieved_user.education_level == user.education_level
    assert retrieved_user.years_experience == user.years_experience
    assert retrieved_user.created_at.isoformat() == user.created_at.isoformat()
    assert retrieved_user.updated_at.isoformat() == user.updated_at.isoformat()
    mock_redis_client.get.assert_called_with(f"user:{user.id}")

    # Check if user was stored in cache
    user_dict = {
        "id": user.id, "email": user.email, "name": user.name,
        "occupation_code": user.occupation_code, "education_level": user.education_level,
        "years_experience": user.years_experience,
        "created_at": user.created_at.isoformat(),
        "updated_at": user.updated_at.isoformat()
    }
    mock_redis_client.set.assert_called_with(f"user:{user.id}", json.dumps(user_dict), ex=3600)


@pytest.mark.asyncio
async def test_get_by_id_cached_hit(cached_user_repo_fixture: CachedUserRepository, async_session_fixture: AsyncSession, mock_redis_client: mock.AsyncMock, mocker):
    """Tests get_by_id_cached for a cache hit scenario."""
    # Setup: User in cache, mock underlying DB access to ensure it's not called
    user_id = str(uuid.uuid4())
    # Ensure datetime objects are timezone-aware for isoformat
    now_tz = datetime.now(tz=datetime.now().astimezone().tzinfo)
    cached_user_dict = {
        "id": user_id, "email": "cached_hit@example.com", "name": "Cached Hit User",
        "occupation_code": "DEV", "education_level": "Bachelors", "years_experience": 3.0,
        "created_at": now_tz.isoformat(),
        "updated_at": now_tz.isoformat()
    }
    mock_redis_client.get.return_value = json.dumps(cached_user_dict)

    # Spy on super().get_by_id to confirm it's not called
    spy_get_by_id = mocker.patch.object(UserRepository, 'get_by_id', new_callable=mocker.AsyncMock)
    spy_get_by_id.return_value = User(email="db@example.com", name="DB User") # This return value should not be used if cache hits

    # Action
    retrieved_user = await cached_user_repo_fixture.get_by_id_cached(user_id)

    # Assertions
    assert retrieved_user is not None
    assert retrieved_user.id == user_id
    assert retrieved_user.email == "cached_hit@example.com"
    assert retrieved_user.name == "Cached Hit User"
    assert retrieved_user.occupation_code == "DEV"
    assert retrieved_user.education_level == "Bachelors"
    assert retrieved_user.years_experience == 3.0
    assert retrieved_user.created_at.isoformat() == now_tz.isoformat()
    assert retrieved_user.updated_at.isoformat() == now_tz.isoformat()
    mock_redis_client.get.assert_called_once_with(f"user:{user_id}")
    spy_get_by_id.assert_not_called() # Should not hit DB
    mock_redis_client.set.assert_not_called() # Should not set cache if it was a hit


@pytest.mark.asyncio
async def test_get_by_id_cached_db_miss(cached_user_repo_fixture: CachedUserRepository, async_session_fixture: AsyncSession, mock_redis_client: mock.AsyncMock, mocker):
    """Tests get_by_id_cached when user is not found in DB or cache."""
    user_id = str(uuid.uuid4())
    mock_redis_client.get.return_value = None # Cache miss

    # Mock the super call to return None, simulating DB miss
    spy_get_by_id = mocker.patch.object(UserRepository, 'get_by_id', new_callable=mocker.AsyncMock)
    spy_get_by_id.return_value = None

    retrieved_user = await cached_user_repo_fixture.get_by_id_cached(user_id)

    assert retrieved_user is None
    mock_redis_client.get.assert_called_once_with(f"user:{user_id}")
    spy_get_by_id.assert_called_once_with(user_id) # Should attempt to fetch from DB
    mock_redis_client.set.assert_not_called() # Should not cache None


@pytest.mark.asyncio
async def test_get_latest_score_cached_miss_then_db_hit(cached_user_repo_fixture: CachedUserRepository, async_session_fixture: AsyncSession, mock_redis_client: mock.AsyncMock):
    """Tests get_latest_score_cached for a cache miss where score is found in DB."""
    # Setup: User and Score in DB, not in cache
    user = User(email="score_cached_miss@example.com", name="Score Miss User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    score_data = AIRScore(
        id=str(uuid.uuid4()), user_id=user.id, occupation_code="TEST_OCC", air_score=90.0, vr_score=80.0, hr_score=70.0, synergy_score=85.0,
        ci_lower=88.0, ci_upper=92.0, parameter_version="v2.0", calculation_metadata={"algo": "X"}
    )
    async_session_fixture.add(score_data)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(score_data)

    mock_redis_client.get.return_value = None # Ensure cache miss

    # Action
    retrieved_score = await cached_user_repo_fixture.get_latest_score_cached(user.id)

    # Assertions
    assert retrieved_score is not None
    assert retrieved_score.id == score_data.id
    assert retrieved_score.user_id == score_data.user_id
    assert retrieved_score.occupation_code == score_data.occupation_code
    assert retrieved_score.air_score == score_data.air_score
    # All other attributes should match as well
    assert retrieved_score.created_at.isoformat() == score_data.created_at.isoformat()
    assert retrieved_score.updated_at.isoformat() == score_data.updated_at.isoformat()
    mock_redis_client.get.assert_called_with(f"latest_airscore:{user.id}")

    # Check if score was stored in cache
    score_dict = {
        "id": score_data.id, "user_id": score_data.user_id, "occupation_code": score_data.occupation_code,
        "air_score": score_data.air_score, "vr_score": score_data.vr_score, "hr_score": score_data.hr_score, "synergy_score": score_data.synergy_score,
        "ci_lower": score_data.ci_lower, "ci_upper": score_data.ci_upper,
        "parameter_version": score_data.parameter_version,
        "calculation_metadata": score_data.calculation_metadata,
        "created_at": score_data.created_at.isoformat(),
        "updated_at": score_data.updated_at.isoformat()
    }
    mock_redis_client.set.assert_called_with(f"latest_airscore:{user.id}", json.dumps(score_dict), ex=300)


@pytest.mark.asyncio
async def test_get_latest_score_cached_hit(cached_user_repo_fixture: CachedUserRepository, async_session_fixture: AsyncSession, mock_redis_client: mock.AsyncMock, mocker):
    """Tests get_latest_score_cached for a cache hit scenario."""
    # Setup: Score in cache, mock underlying DB access to ensure it's not called
    user_id = str(uuid.uuid4())
    now_tz = datetime.now(tz=datetime.now().astimezone().tzinfo)
    cached_score_dict = {
        "id": str(uuid.uuid4()), "user_id": user_id, "occupation_code": "CACHED_OCC",
        "air_score": 95.5, "vr_score": 0.8, "hr_score": 0.9, "synergy_score": 0.92,
        "ci_lower": 94.0, "ci_upper": 96.0,
        "parameter_version": "v2.1",
        "calculation_metadata": {"algo": "Y", "metrics": ["perf"]},
        "created_at": now_tz.isoformat(),
        "updated_at": now_tz.isoformat()
    }
    mock_redis_client.get.return_value = json.dumps(cached_score_dict)

    # Spy on super().get_latest_score to confirm it's not called
    spy_get_latest_score = mocker.patch.object(UserRepository, 'get_latest_score', new_callable=mocker.AsyncMock)
    spy_get_latest_score.return_value = None # This return value should not be used if cache hits

    # Action
    retrieved_score = await cached_user_repo_fixture.get_latest_score_cached(user_id)

    # Assertions
    assert retrieved_score is not None
    assert retrieved_score.user_id == user_id
    assert retrieved_score.air_score == 95.5
    assert retrieved_score.calculation_metadata == {"algo": "Y", "metrics": ["perf"]}
    assert retrieved_score.created_at.isoformat() == now_tz.isoformat()
    assert retrieved_score.updated_at.isoformat() == now_tz.isoformat()
    mock_redis_client.get.assert_called_once_with(f"latest_airscore:{user_id}")
    spy_get_latest_score.assert_not_called() # Should not hit DB
    mock_redis_client.set.assert_not_called() # Should not set cache if it was a hit


@pytest.mark.asyncio
async def test_get_latest_score_cached_db_miss(cached_user_repo_fixture: CachedUserRepository, async_session_fixture: AsyncSession, mock_redis_client: mock.AsyncMock, mocker):
    """Tests get_latest_score_cached when no score is found in DB or cache."""
    user = User(email="no_score@example.com", name="No Score User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    mock_redis_client.get.return_value = None # Cache miss

    # Mock the super call to return None, simulating DB miss
    spy_get_latest_score = mocker.patch.object(UserRepository, 'get_latest_score', new_callable=mocker.AsyncMock)
    spy_get_latest_score.return_value = None

    retrieved_score = await cached_user_repo_fixture.get_latest_score_cached(user.id)

    assert retrieved_score is None
    mock_redis_client.get.assert_called_once_with(f"latest_airscore:{user.id}")
    spy_get_latest_score.assert_called_once_with(user.id) # Should attempt to fetch from DB
    mock_redis_client.set.assert_not_called() # Should not cache None


@pytest.mark.asyncio
async def test_invalidate_user_cache(cached_user_repo_fixture: CachedUserRepository, mock_redis_client: mock.AsyncMock):
    """Tests that invalidate_user_cache correctly calls redis.delete for both keys."""
    user_id = str(uuid.uuid4())
    user_key = f"user:{user_id}"
    latest_airscore_key = f"latest_airscore:{user_id}"

    await cached_user_repo_fixture.invalidate_user_cache(user_id)

    mock_redis_client.delete.assert_any_call(user_key)
    mock_redis_client.delete.assert_any_call(latest_airscore_key)
    assert mock_redis_client.delete.call_count == 2 # Ensure both were called.

# Placeholder for get_session that needs to be patched in tests
async def get_session_patchable() -> AsyncSession:
    # This will be replaced by a fixture in actual tests.
    # It must be a generator to match the signature.
    # Yielding a mock session as a fallback if not patched.
    yield mock.AsyncMock(spec=AsyncSession)


class DomainEventRepository:
    """Repository for managing Domain Events."""
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, event: DomainEvent) -> DomainEvent:
        self.session.add(event)
        await self.session.flush() # Flush to get ID if needed, but don't commit here
        return event

    async def get_pending_events(self, limit: int = 100) -> List[DomainEvent]:
        result = await self.session.execute(
            select(DomainEvent)
            .where(DomainEvent.status == "pending")
            .order_by(DomainEvent.created_at)
            .limit(limit)
        )
        return result.scalars().all()

    async def mark_as_published(self, event_ids: List[str]):
        # Corrected to use update() directly on the table, not select().values()
        await self.session.execute(
            update(DomainEvent)
            .where(DomainEvent.id.in_(event_ids))
            .values(status="published", published_at=func.now())
        )
        await self.session.commit() # Commit the status update


# Business service function (simulated) that uses the Outbox Pattern
async def calculate_and_store_airscore(user_id: str, occupation_code: str = "SW_ENG"):
    async for session in get_session_patchable(): # Use patchable generator
        user_repo = UserRepository(session)
        event_repo = DomainEventRepository(session)

        try:
            # 1. Perform business operation: Create AIRScore
            new_score = AIRScore(
                user_id=user_id,
                occupation_code=occupation_code,
                air_score=92.1, vr_score=0.75, hr_score=0.85, synergy_score=0.95,
                ci_lower=0.85, ci_upper=0.98,
                parameter_version="v1.3",
                calculation_metadata={"model": "deep_eval_v4", "factors": ["fluency", "domain", "adaptive"]}
            )
            session.add(new_score)
            await session.flush() # Flush to get the ID for new_score before committing

            # 2. Record corresponding domain event in the Outbox table (same transaction)
            event_payload = {
                "user_id": user_id,
                "air_score_id": new_score.id,
                "score_value": new_score.air_score,
                "occupation": occupation_code
            }
            new_event = DomainEvent(
                event_type=EventType.SCORE_CALCULATED,
                aggregate_type="AIRScore",
                aggregate_id=new_score.id,
                payload=event_payload,
                status="pending"
            )
            await event_repo.add(new_event)

            await session.commit() # Both operations commit atomically
            # Removed print statements for tests
            return new_score, new_event
        except Exception: # Catch all to ensure rollback
            await session.rollback()
            # Removed print statements for tests
            raise

# Simulate an event publisher that polls the outbox table
async def event_publisher(interval_seconds: float = 0.01, max_events: int = 5, stop_condition: asyncio.Event = None):

    # Using a list to store processed events for testing purposes
    processed_events_in_run = []

    try:
        while True:
            if stop_condition and stop_condition.is_set():
                break

            async for session in get_session_patchable(): # Use patchable generator
                event_repo = DomainEventRepository(session)
                pending_events = await event_repo.get_pending_events(limit=max_events)

                if pending_events:
                    event_ids_to_publish = [event.id for event in pending_events]
                    processed_events_in_run.extend(pending_events)

                    await event_repo.mark_as_published(event_ids_to_publish)

            await asyncio.sleep(interval_seconds)

    except asyncio.CancelledError:
        pass # Graceful shutdown
    return processed_events_in_run


# The `run_outbox_demo` and top-level await calls are for demonstration purposes in a notebook,
# and will not be directly tested in unit tests. Individual components are tested.


# --- Pytest Unit Tests ---

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(scope="function")
async def async_session_fixture():
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)

    async with AsyncSessionLocal() as session:
        yield session

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest.mark.asyncio
async def test_domain_event_repository_add(async_session_fixture: AsyncSession):
    """Tests adding a domain event to the repository."""
    event_repo = DomainEventRepository(async_session_fixture)
    test_event = DomainEvent(
        event_type=EventType.USER_CREATED,
        aggregate_type="User",
        aggregate_id=str(uuid.uuid4()),
        payload={"message": "user created"}
    )
    added_event = await event_repo.add(test_event)

    assert added_event is test_event
    assert added_event.id is not None
    assert added_event.status == "pending"

    # Verify it's in the session, but not yet committed
    # Need to commit to verify persistence via a fresh query later
    await async_session_fixture.commit()
    retrieved_event_query = await async_session_fixture.execute(
        select(DomainEvent).where(DomainEvent.id == added_event.id)
    )
    retrieved_event = retrieved_event_query.scalar_one_or_none()
    assert retrieved_event.id == added_event.id


@pytest.mark.asyncio
async def test_domain_event_repository_get_pending_events(async_session_fixture: AsyncSession):
    """Tests retrieving pending domain events, respecting order and limit."""
    event_repo = DomainEventRepository(async_session_fixture)
    user_id = str(uuid.uuid4())

    # Create events with different timestamps
    # Need to ensure created_at values are distinct enough for order_by in SQLite
    # Use timezone-aware datetimes for consistency with mapped_column(DateTime(timezone=True))
    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    event1 = DomainEvent(event_type=EventType.USER_CREATED, aggregate_type="User", aggregate_id=user_id, payload={}, created_at=now - timedelta(seconds=10)) # Oldest
    event2 = DomainEvent(event_type=EventType.ASSESSMENT_COMPLETED, aggregate_type="Assessment", aggregate_id=str(uuid.uuid4()), payload={}, created_at=now - timedelta(seconds=5))
    event3 = DomainEvent(event_type=EventType.SCORE_CALCULATED, aggregate_type="AIRScore", aggregate_id=str(uuid.uuid4()), payload={}, created_at=now - timedelta(seconds=2))

    # Add a published event that should not be returned by get_pending_events
    published_event = DomainEvent(event_type=EventType.USER_UPDATED, aggregate_type="User", aggregate_id=user_id, payload={}, status="published", created_at=now - timedelta(seconds=1))

    async_session_fixture.add_all([event1, event2, event3, published_event])
    await async_session_fixture.commit()

    # Test limit and order
    pending_events = await event_repo.get_pending_events(limit=2)
    assert len(pending_events) == 2
    assert pending_events[0].id == event1.id # Oldest first
    assert pending_events[1].id == event2.id

    # Test with no limit
    all_pending_events = await event_repo.get_pending_events()
    assert len(all_pending_events) == 3
    assert all_pending_events[0].id == event1.id
    assert all_pending_events[1].id == event2.id
    assert all_pending_events[2].id == event3.id

    # Test with no pending events after marking all as published
    await async_session_fixture.execute(
        update(DomainEvent)
        .where(DomainEvent.status == "pending")
        .values(status="published", published_at=func.now())
    )
    await async_session_fixture.commit()
    no_pending_events = await event_repo.get_pending_events()
    assert len(no_pending_events) == 0


@pytest.mark.asyncio
async def test_domain_event_repository_mark_as_published(async_session_fixture: AsyncSession):
    """Tests marking events as published."""
    event_repo = DomainEventRepository(async_session_fixture)
    event1 = DomainEvent(event_type=EventType.USER_CREATED, aggregate_type="User", aggregate_id=str(uuid.uuid4()), payload={})
    event2 = DomainEvent(event_type=EventType.ASSESSMENT_COMPLETED, aggregate_type="Assessment", aggregate_id=str(uuid.uuid4()), payload={})

    async_session_fixture.add_all([event1, event2])
    await async_session_fixture.commit()

    # Ensure they are pending initially
    retrieved_events = await event_repo.get_pending_events()
    assert len(retrieved_events) == 2
    assert all(e.status == "pending" for e in retrieved_events)

    # Mark one event as published
    await event_repo.mark_as_published([event1.id])

    # Verify status in DB
    updated_event_query = await async_session_fixture.execute(
        select(DomainEvent).where(DomainEvent.id == event1.id)
    )
    updated_event = updated_event_query.scalar_one_or_none()
    assert updated_event.status == "published"
    assert updated_event.published_at is not None

    # Verify the other event is still pending
    other_event_query = await async_session_fixture.execute(
        select(DomainEvent).where(DomainEvent.id == event2.id)
    )
    other_event = other_event_query.scalar_one_or_none()
    assert other_event.status == "pending"
    assert other_event.published_at is None

    # Mark both as published
    await event_repo.mark_as_published([event1.id, event2.id])
    all_published_events = await event_repo.get_pending_events()
    assert len(all_published_events) == 0


@pytest.mark.asyncio
async def test_calculate_and_store_airscore_success(async_session_fixture: AsyncSession, mocker):
    """
    Tests atomic creation of AIRScore and DomainEvent.
    Patches `get_session_patchable` to use `async_session_fixture`.
    """
    # Patch the get_session_patchable generator to always yield our fixture's session
    mocker.patch(__name__ + '.get_session_patchable', return_value=iter([async_session_fixture]))

    user = User(email="user_for_score@example.com", name="Score User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    score, event = await calculate_and_store_airscore(user.id, "AI_Dev")

    assert score is not None
    assert event is not None
    assert score.user_id == user.id
    assert score.air_score == 92.1
    assert event.aggregate_id == score.id
    assert event.event_type == EventType.SCORE_CALCULATED.value
    assert event.status == "pending"
    assert event.payload["user_id"] == user.id
    assert event.payload["air_score_id"] == score.id
    assert event.payload["score_value"] == score.air_score

    # Verify both are persisted
    retrieved_score_query = await async_session_fixture.execute(select(AIRScore).where(AIRScore.id == score.id))
    retrieved_score = retrieved_score_query.scalar_one_or_none()
    assert retrieved_score.id == score.id

    retrieved_event_query = await async_session_fixture.execute(select(DomainEvent).where(DomainEvent.id == event.id))
    retrieved_event = retrieved_event_query.scalar_one_or_none()
    assert retrieved_event.id == event.id


@pytest.mark.asyncio
async def test_calculate_and_store_airscore_rollback_on_failure(async_session_fixture: AsyncSession, mocker):
    """
    Tests that if an error occurs during score or event creation, the transaction is rolled back.
    """
    # Patch get_session_patchable to inject our fixture's session
    mocker.patch(__name__ + '.get_session_patchable', return_value=iter([async_session_fixture]))

    user = User(email="user_for_rollback@example.com", name="Rollback User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    # Simulate an error during event creation by patching DomainEventRepository.add
    mocker.patch.object(
        DomainEventRepository,
        'add',
        side_effect=ValueError("Simulated event creation error")
    )

    with pytest.raises(ValueError, match="Simulated event creation error"):
        await calculate_and_store_airscore(user.id, "Faulty_Service")

    # Verify no score or event was persisted by querying the session/DB
    all_scores_query = await async_session_fixture.execute(select(AIRScore).where(AIRScore.user_id == user.id))
    assert all_scores_query.scalars().all() == []

    all_events_query = await async_session_fixture.execute(select(DomainEvent).where(DomainEvent.aggregate_id == user.id))
    assert all_events_query.scalars().all() == []


@pytest.mark.asyncio
async def test_event_publisher_publishes_events(async_session_fixture: AsyncSession, mocker):
    """
    Tests that the event publisher correctly processes and marks pending events.
    """
    # Patch asyncio.sleep to prevent actual delays and control test flow
    mock_sleep = mocker.patch('asyncio.sleep', new_callable=mocker.AsyncMock)

    # Patch get_session_patchable to continuously yield our fixture's session for multiple cycles
    # Use a list to simulate multiple session yields over publisher's loop
    sessions_to_yield = [async_session_fixture, async_session_fixture, async_session_fixture] # Allow 3 polls
    mocker.patch(__name__ + '.get_session_patchable', return_value=(s for s in sessions_to_yield))

    user = User(email="pub_user@example.com", name="Publisher Test User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    # Create multiple pending events
    events_to_create = []
    now = datetime.now(tz=datetime.now().astimezone().tzinfo) # Ensure timezone aware for comparison
    for i in range(3):
        score_id = str(uuid.uuid4())
        event = DomainEvent(
            event_type=EventType.SCORE_CALCULATED.value,
            aggregate_type="AIRScore",
            aggregate_id=score_id,
            payload={"score_id": score_id, "index": i},
            created_at=now + timedelta(milliseconds=i*10) # Ensure distinct timestamps for ordering
        )
        events_to_create.append(event)

    async_session_fixture.add_all(events_to_create)
    await async_session_fixture.commit() # Commit pending events

    # Use an asyncio.Event to stop the publisher gracefully
    stop_event = asyncio.Event()

    # Start the publisher in a background task
    publisher_task = asyncio.create_task(event_publisher(interval_seconds=0.01, max_events=2, stop_condition=stop_event))

    # Allow publisher to run for a few cycles. Each call to `asyncio.sleep` will yield control.
    # The patched `get_session_patchable` will run out of sessions, implicitly stopping the publisher.
    await publisher_task # Wait for the publisher to finish (it will iterate through the yielded sessions)

    # Verify events are marked as published in the database
    all_events_query = await async_session_fixture.execute(select(DomainEvent).order_by(DomainEvent.created_at))
    all_events = all_events_query.scalars().all()

    assert len(all_events) == 3
    assert all(event.status == "published" for event in all_events)
    assert all(event.published_at is not None for event in all_events)

    # Verify the order of processing (oldest first due to order_by in get_pending_events)
    assert all_events[0].id == events_to_create[0].id
    assert all_events[1].id == events_to_create[1].id
    assert all_events[2].id == events_to_create[2].id

    # Verify sleep was called multiple times, indicating polling loops
    # The patched get_session_patchable provided 3 sessions, so it should loop at least 3 times plus once more to find empty.
    assert mock_sleep.call_count >= 3 # At least 3 sleeps for 3 rounds of polling


@pytest.mark.asyncio
async def test_event_publisher_no_pending_events_stops_early(async_session_fixture: AsyncSession, mocker):
    """
    Tests event publisher behavior when there are no pending events from the start.
    It should poll once, find no events, and then the stop_condition should stop it.
    """
    mock_sleep = mocker.patch('asyncio.sleep', new_callable=mocker.AsyncMock)

    # Patch get_session_patchable to yield our fixture's session for the initial poll
    mocker.patch(__name__ + '.get_session_patchable', return_value=iter([async_session_fixture]))

    stop_event = asyncio.Event()

    # The publisher will run once, find no events, sleep, then loop again.
    # In the second loop, get_pending_events would be empty, then it would check stop_condition.
    # To ensure it stops *after* checking, we'll set the stop event after the first poll.

    # Start the publisher in a background task
    publisher_task = asyncio.create_task(event_publisher(interval_seconds=0.01, max_events=5, stop_condition=stop_event))

    # Allow publisher to run for its first check
    await asyncio.sleep(0.01)

    # Now set the stop condition, it should terminate in the next loop iteration
    stop_event.set()

    # Wait for the publisher to finish
    processed_events = await publisher_task

    assert processed_events == [] # No events were processed

    # Verify no events were unexpectedly created or modified in DB
    all_events_query = await async_session_fixture.execute(select(DomainEvent))
    assert all_events_query.scalars().all() == []

    # Sleep should have been called at least once (after the first empty poll)
    assert mock_sleep.called


@pytest.mark.asyncio
async def test_event_publisher_max_events_limit(async_session_fixture: AsyncSession, mocker):
    """
    Tests that the event publisher respects the max_events limit per polling cycle
    and processes all events across multiple cycles.
    """
    mock_sleep = mocker.patch('asyncio.sleep', new_callable=mocker.AsyncMock)

    # Create a list of sessions to control when the publisher gets a session,
    # simulating multiple polling cycles.
    # We want 3 cycles: 2 events, 2 events, 1 event (then stop)
    sessions_for_publisher = [async_session_fixture] * 4 # 4 sessions for 4 potential polls

    # Patch get_session_patchable to yield sessions from our controlled list
    session_generator = (s for s in sessions_for_publisher)
    mocker.patch(__name__ + '.get_session_patchable', return_value=session_generator)

    user = User(email="limit_user@example.com", name="Limit Test User")
    async_session_fixture.add(user)
    await async_session_fixture.commit()
    await async_session_fixture.refresh(user)

    # Create 5 pending events
    events_to_create = []
    now = datetime.now(tz=datetime.now().astimezone().tzinfo)
    for i in range(5):
        score_id = str(uuid.uuid4())
        event = DomainEvent(
            event_type=EventType.USER_CREATED.value,
            aggregate_type="User",
            aggregate_id=score_id,
            payload={"index": i},
            created_at=now + timedelta(microseconds=i) # Ensure precise ordering for SQLite
        )
        events_to_create.append(event)

    async_session_fixture.add_all(events_to_create)
    await async_session_fixture.commit() # Commit all 5 pending events

    stop_event = asyncio.Event()

    publisher_task = asyncio.create_task(event_publisher(interval_seconds=0.01, max_events=2, stop_condition=stop_event))

    # Allow the publisher to run. Since `get_session_patchable` is a finite generator,
    # the publisher will eventually exhaust it and then stop.
    processed_events_list = await publisher_task

    # Verify all 5 events were processed
    assert len(processed_events_list) == 5

    # Verify events are marked as published in the database
    all_events_query = await async_session_fixture.execute(select(DomainEvent).order_by(DomainEvent.created_at))
    all_events_in_db = all_events_query.scalars().all()

    assert len(all_events_in_db) == 5
    assert all(event.status == "published" for event in all_events_in_db)
    assert all(event.published_at is not None for event in all_events_in_db)

    # Verify the order matches the creation order
    for i, event in enumerate(all_events_in_db):
        assert event.id == events_to_create[i].id

    # The publisher made 3 polls that found events, and then one more that found no events, triggering stop_condition.
    # Thus, mock_sleep should have been called 4 times.
    assert mock_sleep.call_count == 4