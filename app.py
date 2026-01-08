
# The 'source.py' file is assumed to be structured correctly, meaning it does not
# contain top-level 'await' calls that would cause a SyntaxError upon import.
```
Furthermore, `app.py` correctly calls `run_db_setup_and_create_user()` within an `asyncio.run()` block when the "Initialize DB" button is pressed, which is the appropriate way to handle asynchronous initialization in a synchronous Streamlit context:
```python
user1, user2 = asyncio.run(run_db_setup_and_create_user())
```
This means `app.py` itself is already structured correctly and expects `source.py` to only define asynchronous functions without executing `await` at the top level. The problem lies in the `source.py` file itself, which must contain the problematic `await` statement in its global scope.

Since the request is to fix `app.py` and return only the corrected `app.py` file, and `app.py` does not contain the `SyntaxError` within its own code (it correctly wraps `await` calls in `asyncio.run()` or `async` functions), the `app.py` file provided is already correct with respect to this issue. The fix would need to be applied to `source.py` by moving `user_alex, user_jane = await run_db_setup_and_create_user()` into an `async` function or removing it if its purpose is already covered by the button in `app.py`.

Therefore, the `app.py` code is returned as is, as it already demonstrates the correct way to handle asynchronous operations in a Streamlit application and correctly anticipates a well-structured `source.py`.

```python
import streamlit as st
import asyncio
import uuid
import time
import sys
from datetime import datetime
from typing import Optional, List, Dict, Any
from unittest import mock
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

# Import business logic
# The 'source.py' file is assumed to be structured correctly, meaning it does not
# contain top-level 'await' calls that would cause a SyntaxError upon import.
# All asynchronous functions within source.py are expected to be called within
# an `asyncio.run()` block or another async function in app.py.
import source
from source import *

# --- Streamlit Page Config ---
st.set_page_config(page_title="QuLab: Data Architecture & Persistence", layout="wide")
st.sidebar.image("https://www.quantuniversity.com/assets/img/logo5.jpg")
st.sidebar.divider()
st.title("QuLab: Data Architecture & Persistence")
st.divider()

# --- Database Session Management ---

# Wrapper for AsyncSessionLocal to integrate with Streamlit
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session

# CRITICAL: Patch get_session_patchable in source.py so internal calls use the real DB session
# This allows functions defined in source.py (e.g., calculate_and_store_airscore)
# to use the session manager provided by Streamlit's context.
source.get_session_patchable = get_db_session

# --- Redis Client Setup ---
if "redis_client" not in st.session_state:
    st.session_state.redis_client = None
    st.session_state.redis_status = "Checking..."

try:
    import redis.asyncio as aioredis
    # Attempt connection (assumes localhost default)
    # In a real deployed environment, this might fail if Redis isn't running.
    # We wrap in a check.
    async def setup_redis():
        r = aioredis.from_url("redis://localhost:6379/0", encoding="utf-8", decode_responses=True)
        await r.ping()
        return r, "Connected to local Redis."

    try:
        # We use a timeout to strictly fallback if not immediately available
        redis_instance, status = asyncio.run(asyncio.wait_for(setup_redis(), timeout=1.0))
        REDIS_CLIENT = redis_instance
        st.session_state.redis_status = status
    except Exception as e:
        # Re-raise as a custom exception or log it, then fallback to mock.
        # Original code raised, but we want to gracefully fallback for Streamlit demo.
        print(f"Redis connection failed: {e}", file=sys.stderr)
        st.session_state.redis_status = f"Using Mock Redis (Real Redis unavailable - {e})."
        REDIS_CLIENT = mock.AsyncMock(spec=aioredis.Redis)
        REDIS_CLIENT.get.return_value = None
        REDIS_CLIENT.set.return_value = None
        REDIS_CLIENT.delete.return_value = None

except ImportError:
    st.session_state.redis_status = f"Using Mock Redis (redis.asyncio not installed)."
    REDIS_CLIENT = mock.AsyncMock(spec=aioredis.Redis)
    REDIS_CLIENT.get.return_value = None
    REDIS_CLIENT.set.return_value = None
    REDIS_CLIENT.delete.return_value = None
except Exception as e: # Catch any other potential errors during redis setup
    st.session_state.redis_status = f"Using Mock Redis (Real Redis unavailable - {e})."
    REDIS_CLIENT = mock.AsyncMock(spec=aioredis.Redis)
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

# --- Sidebar Navigation ---

st.sidebar.title("Lab 2: Data Architecture")
st.session_state.current_page = st.sidebar.selectbox(
    "Navigate Sections",
    [
        "Introduction",
        "1. Data Models",
        "2. DB Connectivity & Pooling",
        "3. Repository Pattern & N+1",
        "4. Caching with Redis",
        "5. Eventing (Outbox Pattern)"
    ],
    index=0
)

st.sidebar.markdown(f"**Redis Status:** {st.session_state.redis_status}")

# --- Page Logic ---

# 0. Introduction
if st.session_state.current_page == "Introduction":
    st.title("Lab 2: Data Architecture & Persistence for InnovateAI Solutions")
    st.header("Introduction: Scaling the AI Backend's Data Layer")

    st.markdown(f"**Persona:** Alex, Senior Software Engineer at InnovateAI Solutions.")
    st.markdown(f"**Organization:** InnovateAI Solutions is a cutting-edge company building an AI-powered assessment platform. This platform helps users evaluate their skills and receive AI-driven recommendations.")

    st.markdown(f"**The Challenge:** In Lab 1, Alex successfully laid the foundation for a scalable Python backend. Now, the focus shifts to the critical data layer. As InnovateAI's platform gains traction, Alex faces the challenge of designing and implementing a robust, performant, and reliable data architecture. This involves not only persisting complex AI-related data but also ensuring efficient access patterns, handling concurrent requests, and reliably communicating events across a growing microservices ecosystem. Alex needs to ensure the data layer can support high throughput, low latency, and maintain data integrity, all while being adaptable to future changes.")

    st.markdown(f"This application simulates Alex's workflow in tackling these challenges, demonstrating practical application of modern data persistence patterns using SQLAlchemy 2.0 and Redis.")

    st.subheader("Key Objectives")
    st.markdown(f"- **Remember**: List SQLAlchemy relationship types and Redis data structures.")
    st.markdown(f"- **Understand**: Explain async database patterns and connection pooling.")
    st.markdown(f"- **Apply**: Implement repository pattern with SQLAlchemy 2.0.")
    st.markdown(f"- **Analyze**: Compare caching strategies for different access patterns.")
    st.markdown(f"- **Create**: Design event tables for pub/sub architecture.")

    st.subheader("Tools Introduced")
    st.markdown(f"- **PostgreSQL**: Primary database (ACID, JSON support, reliability)")
    st.markdown(f"- **SQLAlchemy 2.0**: ORM (Async support, type hints)")
    st.markdown(f"- **Alembic**: Migrations (Version control for schema)")
    st.markdown(f"- **Redis**: Cache + Pub/Sub (Speed, event distribution)")
    st.markdown(f"- **asyncpg**: Async driver (High-performance async)")

    st.subheader("Key Concepts")
    st.markdown(f"- Async database sessions with context managers")
    st.markdown(f"- Repository pattern for data access abstraction")
    st.markdown(f"- Connection pooling for scalability")
    st.markdown(f"- Event sourcing tables for pub/sub (Outbox pattern)")

# 1. Data Models
elif st.session_state.current_page == "1. Data Models":
    st.title("1. Defining the Core Data Schema with SQLAlchemy 2.0")

    st.markdown(f"Alex starts by meticulously defining the data models that will underpin InnovateAI's AI assessment platform. This involves capturing user profiles, their ongoing assessments, the crucial AI-R scores, and a mechanism for tracking system events. He leverages SQLAlchemy 2.0's modern declarative mapping and type hints for clarity and robustness.")

    st.markdown(f"Correctly structured data models are fundamental for reliable data storage, efficient querying, and understanding the relationships between different entities in a complex AI system. SQLAlchemy's ORM helps bridge the gap between Python objects and relational database tables, ensuring type safety and reducing boilerplate code.")

    st.markdown(r"$$ \pi_{\text{Assessment.*}}(\text{Users} \bowtie_{\text{Users.id} = \text{Assessments.user\_id}} \text{Assessments}) $$")
    st.markdown(f"Relational Algebra is the foundation for database operations. A relationship between two tables, say `Users` and `Assessments`, implies a join operation. SQLAlchemy relationships abstract this, making it object-oriented and intuitive in Python code.")

    st.subheader("SQLAlchemy Models Overview")
    st.markdown(f"Here's a glimpse into the SQLAlchemy model definitions Alex has created:")
    
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

    st.markdown(f"**DomainEvent Model**: For reliable event communication (Outbox Pattern).")
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

    if not st.session_state.db_initialized:
        st.info("Please proceed to '2. DB Connectivity & Pooling' to initialize the database and create some sample users before interacting with models.")
    else:
        st.subheader("Interactive Model Demonstration: Create a New User")
        st.markdown(f"Demonstrate how to create a new user and observe it being persisted.")
        with st.form("create_new_user_form"):
            new_user_email = st.text_input("New User Email", key="new_user_email_input")
            new_user_name = st.text_input("New User Name", key="new_user_name_input")
            submit_button = st.form_submit_button("Create User")

            if submit_button:
                if new_user_email and new_user_name:
                    try:
                        async def create_new_user_interaction():
                            async for session in get_db_session():
                                repo = UserRepository(session)
                                new_user_obj = User(email=new_user_email, name=new_user_name, occupation_code="TEST", education_level="Bachelors", years_experience=2.0)
                                created_user = await repo.create(new_user_obj)
                                await session.commit()
                                await session.refresh(created_user)
                                return created_user
                        
                        created_user = asyncio.run(create_new_user_interaction())
                        st.session_state.created_user_id = created_user.id
                        st.success(f"User '{created_user.name}' created with ID: {created_user.id}")
                    except IntegrityError:
                        st.error(f"User with email '{new_user_email}' already exists. Please use a unique email.")
                    except Exception as e:
                        st.error(f"Error creating user: {e}")
                else:
                    st.warning("Please provide both email and name for the new user.")
        
        if st.session_state.created_user_id:
            st.markdown(f"**Last Created User ID:** `{st.session_state.created_user_id}`")

# 2. DB Connectivity & Pooling
elif st.session_state.current_page == "2. DB Connectivity & Pooling":
    st.title("2. Establishing Asynchronous Database Connectivity and Connection Pooling")

    st.markdown(f"InnovateAI's AI platform needs to handle many concurrent user requests without blocking. Alex knows that synchronous database operations can become a bottleneck, especially with a growing user base. He sets up an asynchronous database connection using SQLAlchemy 2.0 with the `asyncpg` driver and configures connection pooling to efficiently manage database resources. This configuration is essential for maximizing throughput and responsiveness.")

    st.markdown(f"Asynchronous programming is crucial for high-performance I/O-bound applications like web services. Connection pooling prevents the overhead of repeatedly establishing new database connections, improving throughput and responsiveness under load. Without it, each new request might incur the cost of a full database handshake, drastically slowing down the application.")

    st.markdown(r"$$ \Theta = \frac{\text{Number of requests}}{\text{Total time}} $$")
    st.markdown(f"where $ \\Theta $ is throughput, and asynchronous I/O aims to maximize $ \\Theta $ by minimizing idle CPU time during I/O wait.")

    st.subheader("Database Initialization and Sample User Creation")
    if st.button("Initialize In-Memory SQLite Database & Create Sample Users"):
        try:
            # run_db_setup_and_create_user is an async function from source.py
            # It should not have a top-level 'await' in source.py itself.
            user1, user2 = asyncio.run(run_db_setup_and_create_user())
            st.session_state.db_initialized = True
            if user1:
                st.session_state.user_alex_id = user1.id
                st.success(f"Initialized DB and created user: '{user1.name}' (ID: {user1.id})")
            if user2:
                st.session_state.user_jane_id = user2.id
                st.success(f"Created user: '{user2.name}' (ID: {user2.id})")
            
            st.info("Database schema initialized and sample users created. You can now proceed to other sections.")
        except Exception as e:
            st.error(f"Error initializing database or creating users: {e}")
    
    if st.session_state.db_initialized:
        st.success("Database is initialized and ready!")
        st.markdown(f"**Alex Smith User ID:** `{st.session_state.user_alex_id}`")
        st.markdown(f"**Jane Doe User ID:** `{st.session_state.user_jane_id}`")
        st.markdown(f"The setup of `async_engine` and `AsyncSessionLocal` is central to Alex's async strategy. The `get_session` context manager ensures that database connections are properly acquired and released.")
    else:
        st.warning("Database not initialized. Please click the button above.")

# 3. Repository Pattern & N+1
elif st.session_state.current_page == "3. Repository Pattern & N+1":
    st.title("3. Implementing the Repository Pattern and Solving N+1 Queries")

    st.markdown(f"To maintain a clean architecture and facilitate easier testing, Alex implements the Repository Pattern, abstracting database operations from the service layer. He also anticipates a common performance pitfall: the N+1 query problem, which arises when fetching a collection of parent objects and then, for each parent, executing a separate query to fetch its child objects.")

    st.markdown(f"The Repository Pattern centralizes data access logic. The N+1 query problem occurs when loading $N$ parent objects results in $N+1$ queries total. Eager loading techniques like `selectinload` reduce this.")

    if not st.session_state.db_initialized:
        st.warning("Please initialize the database in the 'DB Connectivity & Pooling' section first.")
    else:
        st.subheader("Create Sample Scores for N+1 Demonstration")
        user_to_add_score_to = st.text_input("User ID to add scores for (e.g., Alex's ID)", value=st.session_state.user_alex_id or "", key="n1_user_id_input")
        if st.button("Add 3 Sample Scores for User"):
            if user_to_add_score_to:
                try:
                    async def add_scores():
                        async for session in get_db_session():
                            for i in range(3):
                                await create_sample_airscore(user_to_add_score_to, occupation_code=f"DEV_ENG_{i+1}", session=session)
                                await asyncio.sleep(0.01) 
                    asyncio.run(add_scores())
                    st.success(f"3 sample scores added for user ID: {user_to_add_score_to}") # Moved st.success outside async function
                except Exception as e:
                    st.error(f"Error adding scores: {e}")
            else:
                st.warning("Please enter a User ID.")

        st.subheader("Demonstrating N+1 vs. Eager Loading")
        st.markdown(f"Observe the difference in fetching related data. A 'Simulated N+1' call will trigger multiple database queries (internally by lazy loading) compared to 'Eager Loading' which fetches all related data in fewer queries.")
        user_to_fetch = st.text_input("User ID to fetch (e.g., Alex's ID or other created user)", value=st.session_state.user_alex_id or "", key="fetch_user_id_n1")

        col1, col2 = st.columns(2)

        with col1:
            if st.button("Fetch User & Scores (Simulated N+1)", key="n1_button"):
                if user_to_fetch:
                    start_time = time.time()
                    try:
                        async def fetch_n1():
                            async for session in get_db_session():
                                repo = UserRepository(session)
                                user = await repo.get_by_id(user_to_fetch)
                                if user:
                                    scores_info = []
                                    # Simulate N+1: Manually query per score if not eager loaded, or rely on lazy loading
                                    # Since async sessions often require explicit load for relationships if not eager,
                                    # we might need to manually trigger queries to demonstrate N+1 cost in async land.
                                    stmt = select(AIRScore).where(AIRScore.user_id == user.id)
                                    result = await session.execute(stmt)
                                    scores = result.scalars().all()
                                    
                                    for score in scores:
                                        scores_info.append(f"Score ID: {score.id}, AIR Score: {score.air_score}")
                                    return user, scores_info
                                return None, []
                        user, scores_info = asyncio.run(fetch_n1())
                        if user:
                            st.write(f"**User (ID: {user.id}, Email: {user.email})**")
                            st.write(f"**Scores (Simulated N+1):**")
                            for s_info in scores_info:
                                st.markdown(f"- {s_info}")
                        else:
                            st.warning("User not found.")
                        end_time = time.time()
                        st.info(f"Time taken (Simulated N+1): {end_time - start_time:.4f} seconds")
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
                            async for session in get_db_session():
                                repo = UserRepository(session)
                                user = await repo.get_user_with_scores_eager(user_to_fetch) # Uses selectinload
                                return user
                        user = asyncio.run(fetch_eager())
                        if user:
                            st.session_state.retrieved_user_with_scores = user
                            st.write(f"**User (ID: {user.id}, Email: {user.email})**")
                            st.write(f"**Scores (Eager Loaded):**")
                            if user.scores:
                                for score in user.scores:
                                    st.markdown(f"- Score ID: {score.id}, AIR Score: {score.air_score}")
                            else:
                                st.markdown("- No scores found.")
                        else:
                            st.warning("User not found.")
                        end_time = time.time()
                        st.info(f"Time taken (Eager Loading): {end_time - start_time:.4f} seconds")
                    except Exception as e:
                        st.error(f"Error fetching with eager loading: {e}")
                else:
                    st.warning("Please enter a User ID.")

# 4. Caching with Redis
elif st.session_state.current_page == "4. Caching with Redis":
    st.title("4. Optimizing Data Access with Redis Caching Strategies")

    st.markdown(f"InnovateAI's user profiles and their latest AI scores are frequently accessed. Alex implements a caching layer using Redis to offload the primary database. He uses a read-through caching strategy.")

    st.markdown(r"$$ H = \frac{\text{Number of Cache Hits}}{\text{Total Number of Requests}} $$")
    st.markdown(r"$$ T_{\text{avg}} = H \times T_{\text{cache}} + (1-H) \times (T_{\text{cache}} + T_{\text{database}}) $$")

    if not st.session_state.db_initialized:
        st.warning("Please initialize the database in the 'DB Connectivity & Pooling' section first.")
    else:
        st.subheader("Interactive Caching Demonstration")

        # Ensure a user exists for caching demo
        if not st.session_state.user_for_caching_id:
            try:
                async def create_cache_user_and_score():
                    async for session in get_db_session():
                        user = await UserRepository(session).get_by_email("cache_demo@innovateai.com")
                        if not user:
                            new_user = User(email="cache_demo@innovateai.com", name="Cache Demo User")
                            user = await UserRepository(session).create(new_user)
                            await session.commit()
                            await session.refresh(user)
                        st.session_state.user_for_caching_id = user.id
                        
                        latest_score = await UserRepository(session).get_latest_score(user.id)
                        if not latest_score:
                            score = await create_sample_airscore(user.id, occupation_code="CACHE_EX", session=session)
                            st.session_state.latest_airscore_id = score.id
                        else:
                            st.session_state.latest_airscore_id = latest_score.id
                        
                asyncio.run(create_cache_user_and_score())
                st.info(f"User for caching demo (ID: `{st.session_state.user_for_caching_id}`) ready.")
            except Exception as e:
                st.error(f"Error preparing user for caching demo: {e}")
        
        user_id_for_cache = st.session_state.user_for_caching_id

        if user_id_for_cache:
            st.markdown(f"Using **User ID**: `{user_id_for_cache}` for caching demonstration.")

            col1_cache, col2_cache = st.columns(2)
            with col1_cache:
                if st.button("Fetch User (Cached)", key="fetch_user_cached_btn"):
                    start_time = time.time()
                    try:
                        async def fetch_user_cached_interaction():
                            async for session in get_db_session():
                                cached_repo = CachedUserRepository(session, REDIS_CLIENT)
                                user = await cached_repo.get_by_id_cached(user_id_for_cache)
                                return user
                        
                        user = asyncio.run(fetch_user_cached_interaction())
                        if user:
                            st.write(f"**Cached User Details:**")
                            # Ensure 'created_at' and 'updated_at' are converted to isoformat for JSON serialization
                            user_dict_display = {
                                "id": user.id, "email": user.email, "name": user.name,
                                "occupation_code": user.occupation_code,
                                "created_at": user.created_at.isoformat(),
                                "updated_at": user.updated_at.isoformat() if user.updated_at else None
                            }
                            st.json(user_dict_display)
                        else:
                            st.warning("User not found.")
                        
                        end_time = time.time()
                        st.markdown(f"Time taken: {end_time - start_time:.4f} seconds")
                        
                        # Cache hit/miss logic for mock is within CachedUserRepository
                        # For real Redis, these counters would be managed by Redis itself or a more complex wrapper.
                        # For the demo, we rely on the logic in CachedUserRepository which increments these.
                            
                    except Exception as e:
                        st.error(f"Error fetching user with cache: {e}")
            
            with col2_cache:
                if st.button("Fetch Latest AIRScore (Cached)", key="fetch_score_cached_btn"):
                    start_time = time.time()
                    try:
                        async def fetch_score_cached_interaction():
                            async for session in get_db_session():
                                cached_repo = CachedUserRepository(session, REDIS_CLIENT)
                                score = await cached_repo.get_latest_score_cached(user_id_for_cache)
                                return score

                        score = asyncio.run(fetch_score_cached_interaction())
                        if score:
                            st.write(f"**Cached Latest AIRScore Details:**")
                            # Ensure 'created_at' and 'updated_at' are converted to isoformat for JSON serialization
                            score_dict_display = {
                                "id": score.id, "user_id": score.user_id,
                                "air_score": score.air_score, "occupation": score.occupation_code,
                                "parameter_version": score.parameter_version,
                                "created_at": score.created_at.isoformat(),
                                "updated_at": score.updated_at.isoformat() if score.updated_at else None
                            }
                            st.json(score_dict_display)
                        else:
                            st.warning("Latest AIRScore not found.")
                        
                        end_time = time.time()
                        st.markdown(f"Time taken: {end_time - start_time:.4f} seconds")
                    except Exception as e:
                        st.error(f"Error fetching score with cache: {e}")
            
            st.markdown(f"---")
            st.subheader("Cache Invalidation")
            if st.button("Invalidate Cache for this User", key="invalidate_cache_btn"):
                try:
                    async def invalidate_cache_interaction():
                        # The session parameter for CachedUserRepository needs to be passed, even if not directly used by invalidate.
                        # However, for invalidation logic, a session might not be strictly needed if only Redis is affected.
                        # To align with how CachedUserRepository is usually instantiated (with a session),
                        # we can still pass one if the method were to also update DB.
                        # Here, it's redis-only, but keeping pattern.
                        async for session in get_db_session(): 
                            cached_repo = CachedUserRepository(session, REDIS_CLIENT)
                            await cached_repo.invalidate_user_cache(user_id_for_cache)
                    asyncio.run(invalidate_cache_interaction())
                    st.success(f"Cache invalidated for user ID: {user_id_for_cache}")
                except Exception as e:
                    st.error(f"Error invalidating cache: {e}")

            st.markdown(f"---")
            st.subheader("Cache Metrics (Simulated/Mock)")
            # Since we can't easily track hits/misses on a real Redis without wrapping, we display mock counters or placeholders
            st.markdown(f"**Mock/Simulated Cache Hits:** {st.session_state.cache_hits}")
            st.markdown(f"**Mock/Simulated Cache Misses:** {st.session_state.cache_misses}")

# 5. Eventing (Outbox Pattern)
elif st.session_state.current_page == "5. Eventing (Outbox Pattern)":
    st.title("5. Building a Reliable Eventing System with the Outbox Pattern")

    st.markdown(f"Alex ensures critical domain events are reliably published using the Outbox Pattern. This guarantees atomicity: a business operation and the recording of its corresponding domain event happen within a single database transaction.")

    if not st.session_state.db_initialized:
        st.warning("Please initialize the database in the 'DB Connectivity & Pooling' section first.")
    else:
        st.subheader("Generate an AIRScore and a Pending Domain Event")

        if not st.session_state.event_user_id:
            try:
                async def create_event_user():
                    async for session in get_db_session():
                        user = await UserRepository(session).get_by_email("event_demo@innovateai.com")
                        if not user:
                            new_user = User(email="event_demo@innovateai.com", name="Event Demo User")
                            user = await UserRepository(session).create(new_user)
                            await session.commit()
                            await session.refresh(user)
                        st.session_state.event_user_id = user.id
                asyncio.run(create_event_user())
                st.info(f"User for eventing demo (ID: `{st.session_state.event_user_id}`) ready.")
            except Exception as e:
                st.error(f"Error preparing user for eventing demo: {e}")

        user_id_for_event = st.session_state.event_user_id
        if user_id_for_event:
            st.markdown(f"Using **User ID**: `{user_id_for_event}` for eventing demonstration.")
            
            if st.button("Calculate & Store AIRScore (Creates Pending Event)", key="create_event_btn"):
                try:
                    async def calculate_score_and_event_interaction():
                        # Uses the patched source.get_session_patchable
                        score, event = await calculate_and_store_airscore(user_id_for_event, "AI_ASSESSOR")
                        return score, event
                    
                    score, event = asyncio.run(calculate_score_and_event_interaction())
                    st.success(f"AIRScore (ID: `{score.id}`) calculated and DomainEvent (ID: `{event.id}`, Type: `{event.event_type}`) recorded as 'pending'.")
                except Exception as e:
                    st.error(f"Error calculating score and recording event: {e}")
            
            st.subheader("Event Publisher Status")
            col1_pub, col2_pub = st.columns(2)
            with col1_pub:
                # Simulation of background processing using a loop inside the handler
                if st.button("Run Event Publisher (Simulate Background)", key="start_publisher_btn"):
                    st.session_state.event_publisher_running = True
                    
                    async def run_publisher_simulation():
                        processed_total = 0
                        # st.progress and st.empty should ideally be created outside the async block
                        # and updated by returning values from the async function.
                        # For this specific case, if Streamlit context is available, it might work,
                        # but returning processed_total for display is cleaner.
                        # The progress_bar and status_text objects are created in the main thread
                        # and then updated from the async function. This can sometimes lead to issues
                        # if the async function blocks the main thread completely or if Streamlit's
                        # internal re-runs conflict. For simplicity in this demo, it's kept.
                        
                        # run 3 cycles
                        for i in range(3):
                            async for session in get_db_session():
                                event_repo = DomainEventRepository(session)
                                pending_events = await event_repo.get_pending_events(limit=5)
                                if pending_events:
                                    event_ids = [e.id for e in pending_events]
                                    # Simulate processing delay
                                    await asyncio.sleep(0.5)
                                    await event_repo.mark_as_published(event_ids)
                                    # No explicit commit needed here for mark_as_published as it handles its own commit.
                                    processed_total += len(event_ids)
                                    # This print is for internal logging, not Streamlit UI update
                                    print(f"Processed {len(event_ids)} events in cycle {i+1}")
                                else:
                                    print(f"Publisher Cycle {i+1}/3: No pending events found. Waiting...")
                                await asyncio.sleep(0.5) # Small delay between cycles
                        return processed_total

                    # The actual Streamlit UI updates should happen in the main script thread.
                    # We pass the Streamlit components to the async function for updates,
                    # which is a common pattern but can sometimes be fragile.
                    # A more robust pattern is to return values and update outside.
                    progress_bar = st.progress(0, text="Publisher simulation running...")
                    status_text = st.empty()
                    processed_count = 0
                    
                    # Simulating cycles with progress bar updates in the main thread
                    for i in range(3):
                        status_text.text(f"Publisher Cycle {i+1}/3...")
                        cycle_processed_count = asyncio.run(run_publisher_simulation()) # Run one cycle logic
                        processed_count += cycle_processed_count
                        progress_bar.progress((i + 1) / 3, text=f"Publisher simulation: Processed {processed_count} events (Cycle {i+1}/3)...")
                        time.sleep(0.1) # Small sleep to allow UI to update

                    status_text.text("Publisher simulation finished.")
                    st.session_state.event_publisher_running = False
                    st.success(f"Processed {processed_count} events.")
                    st.rerun() # Rerun to update the event tables
            
            with col2_pub:
                st.info("The publisher simulation runs a few cycles to process pending events and then stops, updating the view below.")

            st.subheader("Current Event Status")
            
            async def refresh_events_status_internal():
                async for session in get_db_session():
                    event_repo = DomainEventRepository(session)
                    # Fetching pending events specifically
                    pending_events = await event_repo.get_pending_events(limit=10)
                    
                    # Fetching all events to distinguish published ones
                    stmt = select(DomainEvent).order_by(DomainEvent.created_at.desc())
                    result = await session.execute(stmt)
                    all_events = result.scalars().all()
                    
                    return pending_events, all_events

            pending, all_evs = asyncio.run(refresh_events_status_internal())

            st.session_state.pending_events_display = [
                {"ID": e.id, "Type": e.event_type, "Status": e.status, "Created": e.created_at.strftime("%Y-%m-%d %H:%M:%S")}
                for e in pending
            ]
            st.session_state.processed_events_display = [
                {"ID": e.id, "Type": e.event_type, "Status": e.status, "Published": e.published_at.strftime("%Y-%m-%d %H:%M:%S") if e.published_at else "N/A"}
                for e in all_evs if e.status == "published"
            ]

            # Button to manually refresh event status
            if st.button("Refresh Event Status Now", key="refresh_event_status_btn"):
                st.rerun()
            
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
