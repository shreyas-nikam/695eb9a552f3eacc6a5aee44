
# Lab 2: Data Architecture & Persistence for InnovateAI Solutions

## Introduction: Scaling the AI Backend's Data Layer

**Persona:** Alex, Senior Software Engineer at InnovateAI Solutions.

**Organization:** InnovateAI Solutions is a cutting-edge company building an AI-powered assessment platform. This platform helps users evaluate their skills and receive AI-driven recommendations.

**The Challenge:** In Lab 1, Alex successfully laid the foundation for a scalable Python backend. Now, the focus shifts to the critical data layer. As InnovateAI's platform gains traction, Alex faces the challenge of designing and implementing a robust, performant, and reliable data architecture. This involves not only persisting complex AI-related data but also ensuring efficient access patterns, handling concurrent requests, and reliably communicating events across a growing microservices ecosystem. Alex needs to ensure the data layer can support high throughput, low latency, and maintain data integrity, all while being adaptable to future changes.

This notebook simulates Alex's workflow in tackling these challenges, demonstrating practical application of modern data persistence patterns using SQLAlchemy 2.0 and Redis.

## 1. Install Required Libraries

Alex begins by ensuring all necessary Python libraries for database interaction, asynchronous operations, and caching are installed.

```python
!pip install sqlalchemy asyncpg redis pydantic uuid
```

## 2. Import Required Dependencies

Next, Alex imports all the modules and classes needed for defining models, setting up asynchronous database connections, and interacting with Redis.

```python
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
```

## 3. Defining the Core Data Schema with SQLAlchemy 2.0

### Story + Context + Real-World Relevance

Alex starts by meticulously defining the data models that will underpin InnovateAI's AI assessment platform. This involves capturing user profiles, their ongoing assessments, the crucial AI-R scores, and a mechanism for tracking system events. He leverages SQLAlchemy 2.0's modern declarative mapping and type hints for clarity and robustness.

Correctly structured data models are fundamental for reliable data storage, efficient querying, and understanding the relationships between different entities in a complex AI system. SQLAlchemy's ORM helps bridge the gap between Python objects and relational database tables, ensuring type safety and reducing boilerplate code.

Relational Algebra is the foundation for database operations. A relationship between two tables, say `Users` and `Assessments`, implies a join operation. For example, to find all assessments for a user, one might perform a projection and join operation: $$ \pi_{\text{Assessment.*}}(\text{Users} \bowtie_{\text{Users.id} = \text{Assessments.user\_id}} \text{Assessments}) $$ SQLAlchemy relationships abstract this, making it object-oriented and intuitive in Python code.

### Code cell (function definition + function execution)

Alex defines the `Base` class for declarative models and a `TimestampMixin` for common `created_at` and `updated_at` fields. Then, he implements the `User`, `Assessment`, `AIRScore`, and `DomainEvent` models exactly as designed, including their relationships and column types.

```python
# Base class for declarative models
Base = declarative_base()

# Mixin for common timestamp fields
class TimestampMixin:
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now())
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=func.now(), onupdate=func.now())

# User model
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

# Assessment model
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
    # responses: Mapped[List["Response"]] = relationship(back_populates="assessment") # Assuming Response model exists, but not defined here for brevity

    def __repr__(self):
        return f"<Assessment(id='{self.id}', user_id='{self.user_id}', status='{self.status}')>"

# AIRScore model
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

# Event Type Enum for Domain Events
class EventType(str, Enum):
    SCORE_CALCULATED = "score.calculated"
    ASSESSMENT_COMPLETED = "assessment.completed"
    PATHWAY_RECOMMENDED = "pathway.recommended"
    BATCH_COMPLETED = "batch.completed"
    USER_CREATED = "user.created"
    USER_UPDATED = "user.updated"

# Domain Event model for Outbox Pattern
class DomainEvent(Base, TimestampMixin):
    __tablename__ = "domain_events"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_type: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    aggregate_type: Mapped[str] = mapped_column(String(100), nullable=False)
    aggregate_id: Mapped[str] = mapped_column(String(36), nullable=False, index=True)
    payload: Mapped[Dict[str, Any]] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="pending") # pending, published, failed
    published_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    __table_args__ = (
        Index('ix_events_status_created', 'status', 'created_at'),
    )

    def __repr__(self):
        return f"<DomainEvent(id='{self.id}', type='{self.event_type}', status='{self.status}')>"

print("SQLAlchemy models (User, Assessment, AIRScore, DomainEvent) defined.")
```

### Markdown cell (explanation of execution)

Alex has now laid out the blueprints for the application's data.
- The `User` model captures core profile information.
- `Assessment` tracks user progress through evaluation sessions.
- `AIRScore` stores the crucial AI-generated evaluation results, including various score components and confidence intervals.
- `DomainEvent` is a special table designed for reliable event communication, which will be explored later.
The relationships like `User` having many `Assessments` and `AIRScore`s are explicitly defined using `relationship` and `ForeignKey`, ensuring data integrity and allowing intuitive navigation between related objects in Python. The `cascade="all, delete-orphan"` on the `User` relationships ensures that related assessments and scores are automatically managed (deleted) when a user is removed, simplifying data lifecycle management for Alex.

## 4. Establishing Asynchronous Database Connectivity and Connection Pooling

### Story + Context + Real-World Relevance

InnovateAI's AI platform needs to handle many concurrent user requests without blocking. Alex knows that synchronous database operations can become a bottleneck, especially with a growing user base. He sets up an asynchronous database connection using SQLAlchemy 2.0 with the `asyncpg` driver and configures connection pooling to efficiently manage database resources. This configuration is essential for maximizing throughput and responsiveness.

Asynchronous programming is crucial for high-performance I/O-bound applications like web services. Connection pooling prevents the overhead of repeatedly establishing new database connections, improving throughput and responsiveness under load. Without it, each new request might incur the cost of a full database handshake, drastically slowing down the application.

Connection Pool Efficiency: The effective connection pool size can be estimated as the number of available connections $N_{avail}$ out of the maximum pool size $N_{max}$. A well-tuned pool minimizes latency due to connection acquisition and releases resources promptly. Throughput ($ \Theta $), the rate at which requests are processed, is given by $ \Theta = \frac{\text{Number of requests}}{\text{Total time}} $. Asynchronous I/O aims to maximize $ \Theta $ by minimizing idle CPU time during I/O wait.
Furthermore, transactional guarantees (ACID properties) are crucial. Atomicity ensures that operations within a transaction are all or nothing. Consistency guarantees that a transaction brings the database from one valid state to another. Isolation means concurrent transactions produce the same result as if they were executed sequentially. Durability ensures that once a transaction is committed, it remains committed even in case of power loss.

### Code cell (function definition + function execution)

Alex defines a `DATABASE_URL` for PostgreSQL with `asyncpg` and sets up the asynchronous engine and session maker. He then creates an `asynccontextmanager` to provide `AsyncSession` instances, ensuring proper session lifecycle management. Finally, he uses this to create a user, demonstrating a basic async database operation.

```python
# Database URL for PostgreSQL with asyncpg driver
DATABASE_URL = "postgresql+asyncpg://air:air@localhost:5432/air_platform" # Ensure PostgreSQL is running (e.g., via Docker)

# Create the asynchronous engine
# pool_size and max_overflow help configure the connection pool
async_engine = create_async_engine(DATABASE_URL, echo=False, pool_size=10, max_overflow=20)

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
            await session.refresh(new_user)
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
```

### Markdown cell (explanation of execution)

The setup of `async_engine` and `AsyncSessionLocal` is central to Alex's async strategy. The `get_session` context manager ensures that database connections are properly acquired and released, preventing resource leaks and managing transactions implicitly. The `pool_size` and `max_overflow` parameters are crucial for connection pooling, allowing the application to reuse existing connections and handle spikes in demand gracefully, without having to re-establish connections every time.

By observing the output, Alex confirms that the database schema was successfully created and a new user entry was persisted asynchronously. The `IntegrityError` handling demonstrates how Alex ensures data consistency, preventing duplicate user emails. This foundational async setup is critical for the AI platform's scalability.

## 5. Implementing the Repository Pattern and Solving N+1 Queries

### Story + Context + Real-World Relevance

To maintain a clean architecture and facilitate easier testing, Alex implements the Repository Pattern, abstracting database operations from the service layer. He also anticipates a common performance pitfall: the N+1 query problem, which arises when fetching a collection of parent objects and then, for each parent, executing a separate query to fetch its child objects. This can drastically degrade performance, especially when dealing with many related records. Alex addresses this with SQLAlchemy's eager loading techniques.

The Repository Pattern centralizes data access logic, making it easier to manage, test, and potentially swap out ORM or database technologies in the future. The N+1 query problem occurs when loading $N$ parent objects (e.g., users) and then subsequently executing $N$ additional queries to fetch their related child objects (e.g., scores), resulting in $N+1$ queries in total. This is inefficient. Eager loading techniques like `selectinload` reduce this to $1$ or $2$ queries, improving performance significantly.

### Code cell (function definition + function execution)

Alex implements a `UserRepository` with asynchronous CRUD operations and a method to demonstrate eager loading to fix the N+1 query problem. He'll also add a method to create an `AIRScore` to have data for eager loading.

```python
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


# Function to create a sample AIRScore
async def create_sample_airscore(user_id: str, occupation_code: str = "SW_ENG") -> AIRScore:
    async for session in get_session():
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
            print(f"Created AIRScore: {new_score}")
            return new_score
        except Exception as e:
            await session.rollback()
            print(f"Error creating AIRScore: {e}")
            raise

# Demonstrate repository usage and N+1 fix
async def run_repository_demo(user_id: str):
    async for session in get_session():
        user_repo = UserRepository(session)
        
        # 1. Get user by ID
        print("\n--- Demo: Get user by ID ---")
        user = await user_repo.get_by_id(user_id)
        print(f"Found user: {user}")

        # 2. Simulate N+1 problem (conceptually, not logging actual queries here)
        print("\n--- Demo: Simulating N+1 problem (conceptual) ---")
        users_without_eager_load = (await session.execute(select(User))).scalars().all()
        start_time = time.perf_counter()
        for u in users_without_eager_load:
            # This would trigger a separate query for each user's scores if not already loaded
            _ = u.scores
            print(f"Accessing scores for user: {u.name}")
        end_time = time.perf_counter()
        print(f"Time taken for N+1 access: {end_time - start_time:.4f} seconds (conceptual)")


        # 3. Fix N+1 with eager loading
        print("\n--- Demo: Fixing N+1 with eager loading ---")
        start_time = time.perf_counter()
        user_with_scores = await user_repo.get_user_with_scores_eager(user_id)
        if user_with_scores:
            print(f"User '{user_with_scores.name}' with {len(user_with_scores.scores)} scores (eagerly loaded):")
            for score in user_with_scores.scores:
                print(f"  - AIRScore: {score.air_score}")
        end_time = time.perf_counter()
        print(f"Time taken for eager loaded access: {end_time - start_time:.4f} seconds")
        
        print("\n--- Demo: Get all users with eager loaded scores ---")
        all_users = await user_repo.get_all_users_with_scores_eager()
        for u in all_users:
            print(f"User {u.name} has {len(u.scores)} scores.")


# Create some scores for Alex and Jane
if user_alex:
    await create_sample_airscore(user_alex.id)
    await asyncio.sleep(0.01) # Small delay to ensure different timestamps
    await create_sample_airscore(user_alex.id, occupation_code="DATA_SCI")
if user_jane:
    await create_sample_airscore(user_jane.id)

# Run the repository demo for Alex
if user_alex:
    await run_repository_demo(user_alex.id)
```

### Markdown cell (explanation of execution)

By implementing `UserRepository`, Alex has created a clean boundary between the business logic and data access, making the code more modular and testable. The `get_user_with_scores_eager` method, utilizing `selectinload(User.scores)`, directly addresses the N+1 query problem. When retrieving a user, all their associated `AIRScore` objects are fetched in a minimal number of queries (often just two: one for users, one for all related scores across all users using an `IN` clause).

The conceptual demonstration highlights that without eager loading, iterating through `user.scores` for each user could lead to many individual database hits. With eager loading, the related data is brought into memory efficiently in one go, significantly reducing database load and improving response times. Alex observes that the time taken for eager loaded access is typically much lower than the simulated N+1 approach, confirming the performance benefit.

## 6. Optimizing Data Access with Redis Caching Strategies

### Story + Context + Real-World Relevance

InnovateAI's user profiles and their latest AI scores are frequently accessed, especially during the initial loading of the user dashboard. To offload the primary PostgreSQL database and accelerate response times for these hot data points, Alex decides to implement a caching layer using Redis. He needs to consider a read-through caching strategy, where data is fetched from the cache if available, otherwise from the database and then stored in the cache for subsequent requests.

Caching is critical for high-performance applications, reducing latency and database load by storing frequently accessed data in a fast, in-memory store like Redis. The read-through strategy is robust for frequently read, less frequently updated data. It simplifies cache management by encapsulating the cache-or-DB logic.

The effectiveness of caching is measured by the Cache Hit Ratio ($ H $): $$ H = \frac{\text{Number of Cache Hits}}{\text{Total Number of Requests}} $$ A higher $ H $ indicates better cache effectiveness. The Average Access Time ($ T_{avg} $) with caching is given by: $$ T_{avg} = H \times T_{cache} + (1-H) \times (T_{cache} + T_{database}) $$ where $ T_{cache} $ is cache access time and $ T_{database} $ is database access time. A good caching strategy aims to minimize $ T_{avg} $.

### Code cell (function definition + function execution)

Alex sets up an `aioredis` client and implements a `CachedUserRepository` (or enhances the existing one) to include caching for user retrieval and an `AIRScore` endpoint. He will use a read-through caching pattern.

```python
# Redis connection setup
REDIS_URL = "redis://localhost:6379/0" # Ensure Redis is running (e.g., via Docker)
redis_client = aioredis.from_url(REDIS_URL, encoding="utf-8", decode_responses=True)

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
            print(f"Cache HIT for user {user_id}")
            # Deserialize user from JSON string
            user_dict = json.loads(cached_user_data)
            # Create a User object (without deep relationships for simplicity here)
            return User(id=user_dict['id'], email=user_dict['email'], name=user_dict['name'],
                        occupation_code=user_dict['occupation_code'],
                        education_level=user_dict['education_level'],
                        years_experience=user_dict['years_experience'],
                        created_at=datetime.fromisoformat(user_dict['created_at']) if 'created_at' in user_dict else datetime.now(),
                        updated_at=datetime.fromisoformat(user_dict['updated_at']) if 'updated_at' in user_dict else datetime.now()
                        )
        
        print(f"Cache MISS for user {user_id}. Fetching from DB.")
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
            print(f"Cache HIT for latest AIRScore of user {user_id}")
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
        
        print(f"Cache MISS for latest AIRScore of user {user_id}. Fetching from DB.")
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
        print(f"Invalidated cache for user {user_id}")

# Demonstrate caching
async def run_caching_demo(user_id: str):
    async for session in get_session():
        cached_user_repo = CachedUserRepository(session, redis_client)

        print(f"\n--- Caching Demo for user {user_id} ---")

        # First request: Cache MISS
        start_time = time.perf_counter()
        user = await cached_user_repo.get_by_id_cached(user_id)
        end_time = time.perf_counter()
        print(f"Time for first user request: {end_time - start_time:.4f} seconds")
        print(f"Retrieved user: {user.name}")
        
        # Second request: Cache HIT
        start_time = time.perf_counter()
        user = await cached_user_repo.get_by_id_cached(user_id)
        end_time = time.perf_counter()
        print(f"Time for second user request: {end_time - start_time:.4f} seconds")
        print(f"Retrieved user: {user.name}")

        # First request for latest AIRScore: Cache MISS
        start_time = time.perf_counter()
        score = await cached_user_repo.get_latest_score_cached(user_id)
        end_time = time.perf_counter()
        print(f"Time for first AIRScore request: {end_time - start_time:.4f} seconds")
        print(f"Retrieved latest score: {score.air_score if score else 'None'}")

        # Second request for latest AIRScore: Cache HIT
        start_time = time.perf_counter()
        score = await cached_user_repo.get_latest_score_cached(user_id)
        end_time = time.perf_counter()
        print(f"Time for second AIRScore request: {end_time - start_time:.4f} seconds")
        print(f"Retrieved latest score: {score.air_score if score else 'None'}")

        # Invalidate cache and demonstrate miss again
        print("\n--- Invalidate cache and retry ---")
        await cached_user_repo.invalidate_user_cache(user_id)
        user = await cached_user_repo.get_by_id_cached(user_id) # Should be a miss again
        print(f"Retrieved user after invalidation: {user.name}")

# Run caching demo for Alex
if user_alex:
    await run_caching_demo(user_alex.id)

# Close Redis connection when done
await redis_client.close()
print("\nRedis client closed.")
```

### Markdown cell (explanation of execution)

Alex's implementation of `CachedUserRepository` successfully demonstrates a read-through caching strategy. The first request for a user or their latest AIRScore results in a "Cache MISS," indicating the data was fetched from PostgreSQL and then stored in Redis. Subsequent requests for the same data show a "Cache HIT," with significantly faster response times, as the data is retrieved directly from Redis's in-memory store.

The `invalidate_user_cache` method is crucial for maintaining data consistency. When a user's data or score changes, Alex needs to clear the stale cache entries to ensure future requests fetch the freshest data from the database. This approach optimizes data access for read-heavy operations, offloading the primary database and improving the user experience on InnovateAI's platform. Alex can now see a direct performance benefit, crucial for scaling the platform.

## 7. Building a Reliable Eventing System with the Outbox Pattern

### Story + Context + Real-World Relevance

Alex is tasked with ensuring that critical domain events (like an `AIRScore` being calculated or an `Assessment` completing) are reliably published to other microservices within InnovateAI, even in the face of temporary network issues or consumer downtime. He implements the Outbox Pattern, using the `DomainEvent` table as a robust buffer. This pattern is vital for ensuring that services remain loosely coupled and that event-driven architectures maintain data consistency.

In distributed systems, reliable communication between services is paramount. The Outbox Pattern ensures atomicity: a business operation and the recording of its corresponding domain event happen within a single database transaction. This guarantees that events are never lost if the publishing mechanism fails after the business operation succeeds but before the event is sent to the message broker. It leverages the ACID properties of the database. Atomicity ensures the business change and event record are one indivisible unit.

The Outbox Pattern helps achieve **eventual consistency**. In a distributed system, data might not be immediately consistent across all services, but it will eventually converge. By guaranteeing events are eventually delivered, the Outbox pattern facilitates this convergence without requiring a complex two-phase commit protocol across services.

### Code cell (function definition + function execution)

Alex implements a `DomainEventRepository` to manage the `DomainEvent` table. He then demonstrates a business operation (creating an AIRScore) that atomically records a `SCORE_CALCULATED` event in the outbox table. Finally, he simulates a "publisher" process that polls the `DomainEvent` table for pending events and marks them as published.

```python
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
        await self.session.execute(
            select(DomainEvent)
            .where(DomainEvent.id.in_(event_ids))
            .values(status="published", published_at=func.now())
        )
        await self.session.commit() # Commit the status update


# Business service function (simulated) that uses the Outbox Pattern
async def calculate_and_store_airscore(user_id: str, occupation_code: str = "SW_ENG"):
    async for session in get_session():
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
            print(f"\nSuccessfully calculated AIRScore (ID: {new_score.id}) and recorded event (ID: {new_event.id}) atomically.")
            return new_score, new_event
        except Exception as e:
            await session.rollback()
            print(f"\nError in calculate_and_store_airscore: {e}. Transaction rolled back.")
            raise

# Simulate an event publisher that polls the outbox table
async def event_publisher(interval_seconds: int = 2, max_events: int = 5):
    print(f"\n--- Starting simulated event publisher (polls every {interval_seconds}s) ---")
    stop_publisher = False
    try:
        while not stop_publisher:
            async for session in get_session():
                event_repo = DomainEventRepository(session)
                pending_events = await event_repo.get_pending_events(limit=max_events)
                
                if pending_events:
                    print(f"Publisher found {len(pending_events)} pending events:")
                    event_ids_to_publish = []
                    for event in pending_events:
                        print(f"  - Publishing event ID: {event.id}, Type: {event.event_type}, Aggregate: {event.aggregate_type}/{event.aggregate_id}")
                        # In a real system, this is where the event would be sent to Kafka/RabbitMQ
                        event_ids_to_publish.append(event.id)
                    
                    if event_ids_to_publish:
                        await event_repo.mark_as_published(event_ids_to_publish)
                        print(f"Marked {len(event_ids_to_publish)} events as 'published'.")
                else:
                    print("Publisher found no pending events.")
            
            await asyncio.sleep(interval_seconds)
            # For demo, stop after a few cycles
            if len(await DomainEventRepository(AsyncSessionLocal()).get_pending_events()) == 0:
                stop_publisher = True # Stop if no more events are expected for demo
    except asyncio.CancelledError:
        print("Event publisher cancelled.")
    finally:
        print("--- Simulated event publisher stopped ---")


# Run the Outbox Pattern demo
async def run_outbox_demo(user_id: str):
    # Start the publisher in the background
    publisher_task = asyncio.create_task(event_publisher(interval_seconds=1, max_events=2))

    # Perform a business operation that generates an event
    await calculate_and_store_airscore(user_id, "Data_Engineer")
    await asyncio.sleep(0.5) # Give publisher a chance to run
    await calculate_and_store_airscore(user_id, "AI_Researcher")
    await asyncio.sleep(0.5) # Give publisher a chance to run

    # Allow publisher to finish processing
    await publisher_task
    
    # Verify events in DB
    print("\n--- Verifying events in database ---")
    async for session in get_session():
        event_repo = DomainEventRepository(session)
        all_events = (await session.execute(select(DomainEvent))).scalars().all()
        for event in all_events:
            print(f"Event ID: {event.id}, Type: {event.event_type}, Status: {event.status}, Published At: {event.published_at}")

if user_alex:
    await run_outbox_demo(user_alex.id)
```

### Markdown cell (explanation of execution)

Alex's implementation clearly demonstrates the Outbox Pattern. When `calculate_and_store_airscore` is called, both the `AIRScore` record and the `DomainEvent` are created within a single database transaction. This means that either both operations succeed, or both fail and are rolled back, guaranteeing atomicity. The `DomainEvent` is initially marked `pending`.

The simulated `event_publisher` then periodically polls the `domain_events` table for `pending` events. Upon finding them, it simulates publishing them (by printing to console) and then updates their status to `published` in the database. This two-phase commit strategy (business operation + event record in DB, then separate publishing + status update) ensures that no event is ever lost, even if the external message broker is temporarily unavailable. Alex can verify the process by observing the publisher logs and then querying the `DomainEvent` table to see the `status` transition from `pending` to `published`, confirming the reliability of this pattern for InnovateAI's event-driven architecture.
