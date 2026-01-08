
# Streamlit Application Specification: Lab 2: Data Architecture & Persistence

## 1. Application Overview

### Purpose of the Application

This Streamlit application serves as an interactive tutorial and demonstration platform for "Lab 2: Data Architecture & Persistence". It guides Software Developers and Data Engineers through modern Python data management practices using SQLAlchemy 2.0, Redis, and an eventing system. The app allows learners to interact with and observe the core concepts of async database sessions, the repository pattern, connection pooling, caching strategies, and the Outbox pattern for reliable event communication in a simulated real-world AI assessment platform context.

### High-Level Story Flow

The application simulates Alex, a Senior Software Engineer at InnovateAI Solutions, as he tackles data layer challenges for their AI assessment platform.

1.  **Introduction**: Alex starts by reviewing the objectives and challenges for scaling the data layer. This page provides a high-level overview of the lab's goals and the tools involved.
2.  **Data Models**: Alex defines the core data schema using SQLAlchemy 2.0 for `User`, `Assessment`, `AIRScore`, and `DomainEvent` models. The app will visually present these models and their relationships, along with an interactive component to create a new user.
3.  **DB Connectivity & Pooling**: Alex sets up asynchronous database connectivity and configures connection pooling to handle concurrent requests efficiently. The app will allow Alex to initialize an in-memory database and create sample users, demonstrating async operations.
4.  **Repository Pattern & N+1 Queries**: Alex implements the Repository Pattern for clean data access and solves the N+1 query problem using eager loading. The app will demonstrate fetching user data both with and without eager loading to highlight performance differences.
5.  **Caching with Redis**: To optimize access for "hot data", Alex integrates Redis as a caching layer. The app will allow Alex to fetch user and score data, observing simulated cache hits and misses, and demonstrate cache invalidation.
6.  **Eventing (Outbox Pattern)**: Alex builds a reliable eventing system using the Outbox Pattern to ensure critical domain events are consistently published. The app will demonstrate the atomic creation of business data and associated events, and a simulated publisher will process these events from an outbox table.

Throughout the application, `st.session_state` will be used to maintain context, such as user IDs, database initialization status, and the current page, providing a seamless multi-page experience within a single Streamlit script.

## 2. Code Requirements

### Import Statement

```python
import streamlit as st
import asyncio
import uuid
from datetime import datetime
import time # For time measurement in caching/N+1 demos
from typing import Optional, List, Dict, Any # Explicitly import for type hinting in Streamlit context

# IMPORTS FROM source.py
# The instruction states: "The import statement: from source.py import *"
# However, `from source.py import *` is not valid Python syntax in Python modules.
# Assuming `source.py` is a module file in the same directory as `app.py`,
# the correct Python import statement for a runnable application would be `from source import *`.
# We proceed with this interpretation to make the application runnable within Streamlit.
from source import *

# --- Setup for Database and Redis clients ---

# The `async_engine` and `AsyncSessionLocal` objects are defined in `source.py`.
# After `from source import *`, these objects are available globally.

# Wrapper for `AsyncSessionLocal` to integrate with Streamlit and FastAPI-like patterns.
# This generator function ensures that a database session is properly managed.
async def get_db_session():
    async with AsyncSessionLocal() as session:
        yield session

# CRITICAL NOTE ON `source.py` INTERACTION:
# Functions like `calculate_and_store_airscore` and `event_publisher` in `source.py`
# are designed to use `get_session_patchable()`, which, as defined in `source.py`,
# yields a `mock.AsyncMock` for testing purposes. For the Streamlit application to
# function with a *real* database, `source.get_session_patchable` must be
# effectively re-directed to provide a real session (e.g., `get_db_session`).
# This specification assumes that, in the execution environment, `get_session_patchable`
# is appropriately overridden or configured to use `get_db_session` from `app.py`.
# This might involve runtime patching in `app.py` or a global configuration
# if `source.py` cannot be directly modified. For this blueprint, we assume
# this redirection happens without explicit modification of `source.py`'s text
# but through runtime configuration of the `get_session_patchable` function object.
# A simple way to achieve this for a demo could be:
# `source.get_session_patchable = get_db_session` (if source.py were imported as `import source`)
# or by directly re-assigning it after `from source import *` if it's a global function:
globals()['get_session_patchable'] = get_db_session


# Setup for Redis client: try to connect to a real Redis, fallback to mock
try:
    import redis.asyncio as aioredis
    # Attempt to connect to a local Redis instance.
    # The `source.py` file also defines `aioredis` but comments out client creation.
    # We will create it here.
    REDIS_CLIENT = aioredis.from_url("redis://localhost:6379/0", encoding="utf-8", decode_responses=True)
    # Ping to check connection, this is an awaitable call
    async def check_redis_connection():
        await REDIS_CLIENT.ping()
    asyncio.run(check_redis_connection())
    st.session_state.redis_status = "Connected to local Redis."
except Exception as e:
    st.session_state.redis_status = f"Failed to connect to local Redis (Error: {e}). Using a mock client."
    from unittest import mock
    REDIS_CLIENT = mock.AsyncMock(spec=aioredis.Redis)
    # Configure mock client behavior for demo
    REDIS_CLIENT.get.return_value = None # Default: cache miss
    REDIS_CLIENT.set.return_value = None
    REDIS_CLIENT.delete.return_value = None

# Placeholder for event publisher task, which might be started/stopped
# This will be managed in st.session_state
```

### Streamlit Application Structure and Flow

The application will use `st.sidebar.selectbox` for navigation between different sections, simulating a multi-page experience.

#### Session State Initialization

```python
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
    st.session_state.retrieved_user_with_scores = None # Stores User object from eager load demo
if "user_for_caching_id" not in st.session_state:
    st.session_state.user_for_caching_id = None
if "latest_airscore_id" not in st.session_state:
    st.session_state.latest_airscore_id = None
if "cache_hits" not in st.session_state: # Simplified metric for demo
    st.session_state.cache_hits = 0
if "cache_misses" not in st.session_state: # Simplified metric for demo
    st.session_state.cache_misses = 0
if "event_user_id" not in st.session_state:
    st.session_state.event_user_id = None
if "event_publisher_running" not in st.session_state:
    st.session_state.event_publisher_running = False
if "pending_events_display" not in st.session_state:
    st.session_state.pending_events_display = []
if "processed_events_display" not in st.session_state:
    st.session_state.processed_events_display = []
if "publisher_task_id" not in st.session_state: # Store task ID if managing actual background tasks (advanced Streamlit)
    st.session_state.publisher_task_id = None
if "publisher_stop_event" not in st.session_state:
    st.session_state.publisher_stop_event = asyncio.Event()

```

#### Sidebar Navigation

```python
st.sidebar.title("Lab 2: Data Architecture & Persistence")
st.session_state.current_page = st.sidebar.selectbox(
    "Navigate Sections",
    [
        "Introduction",
        "1. Data Models",
        "2. DB Connectivity & Pooling",
        "3. Repository Pattern & N+1",
        "4. Caching with Redis",
        "5. Eventing (Outbox Pattern)"
    ]
)

st.sidebar.markdown(f"**Redis Status:** {st.session_state.redis_status}")
```

#### Page: Introduction

**Content**:

```python
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
```

#### Page: 1. Data Models

**Content**:

```python
elif st.session_state.current_page == "1. Data Models":
    st.title("1. Defining the Core Data Schema with SQLAlchemy 2.0")

    st.markdown(f"Alex starts by meticulously defining the data models that will underpin InnovateAI's AI assessment platform. This involves capturing user profiles, their ongoing assessments, the crucial AI-R scores, and a mechanism for tracking system events. He leverages SQLAlchemy 2.0's modern declarative mapping and type hints for clarity and robustness.")

    st.markdown(f"Correctly structured data models are fundamental for reliable data storage, efficient querying, and understanding the relationships between different entities in a complex AI system. SQLAlchemy's ORM helps bridge the gap between Python objects and relational database tables, ensuring type safety and reducing boilerplate code.")

    st.markdown(r"Relational Algebra is the foundation for database operations. A relationship between two tables, say `Users` and `Assessments`, implies a join operation. For example, to find all assessments for a user, one might perform a projection and join operation: $$ \pi_{{\text{{Assessment.*}}}}(\text{{Users}} \bowtie_{{\text{{Users.id}} = \text{{Assessments.user\_id}}}} \text{{Assessments}}) $$")
    st.markdown(r"where $ \pi $ denotes projection and $ \bowtie $ denotes natural join. SQLAlchemy relationships abstract this, making it object-oriented and intuitive in Python code.")

    st.subheader("SQLAlchemy Models Overview")
    st.markdown(f"Here's a glimpse into the SQLAlchemy model definitions Alex has created:")
    
    st.markdown(f"**User Model**: Represents user profiles.")
    st.code(f"""
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
    st.code(f"""
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
    st.code(f"""
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
    st.code(f"""
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
                                st.session_state.created_user_id = created_user.id
                                st.success(f"User '{created_user.name}' created with ID: {created_user.id}")
                        asyncio.run(create_new_user_interaction())
                    except IntegrityError:
                        st.error(f"User with email '{new_user_email}' already exists. Please use a unique email.")
                    except Exception as e:
                        st.error(f"Error creating user: {e}")
                else:
                    st.warning("Please provide both email and name for the new user.")
        
        if st.session_state.created_user_id:
            st.markdown(f"**Last Created User ID:** `{st.session_state.created_user_id}`")
            st.markdown(f"You can use this ID in subsequent sections.")

```

#### Page: 2. DB Connectivity & Pooling

**Content**:

```python
elif st.session_state.current_page == "2. DB Connectivity & Pooling":
    st.title("2. Establishing Asynchronous Database Connectivity and Connection Pooling")

    st.markdown(f"InnovateAI's AI platform needs to handle many concurrent user requests without blocking. Alex knows that synchronous database operations can become a bottleneck, especially with a growing user base. He sets up an asynchronous database connection using SQLAlchemy 2.0 with the `asyncpg` driver and configures connection pooling to efficiently manage database resources. This configuration is essential for maximizing throughput and responsiveness.")

    st.markdown(f"Asynchronous programming is crucial for high-performance I/O-bound applications like web services. Connection pooling prevents the overhead of repeatedly establishing new database connections, improving throughput and responsiveness under load. Without it, each new request might incur the cost of a full database handshake, drastically slowing down the application.")

    st.markdown(r"Connection Pool Efficiency: The effective connection pool size can be estimated as the number of available connections $N_{{\text{avail}}}$ out of the maximum pool size $N_{{\text{max}}}$. A well-tuned pool minimizes latency due to connection acquisition and releases resources promptly. Throughput ($ \Theta $), the rate at which requests are processed, is given by: $$ \Theta = \frac{{\text{{Number of requests}}}}{{\text{{Total time}}}} $$")
    st.markdown(r"where $ \Theta $ is throughput, and asynchronous I/O aims to maximize $ \Theta $ by minimizing idle CPU time during I/O wait. Furthermore, transactional guarantees (ACID properties) are crucial. Atomicity ensures that operations within a transaction are all or nothing. Consistency guarantees that a transaction brings the database from one valid state to another. Isolation means concurrent transactions produce the same result as if they were executed sequentially. Durability ensures that once a transaction is committed, it remains committed even in case of power loss.")

    st.subheader("Database Initialization and Sample User Creation")
    if st.button("Initialize In-Memory SQLite Database & Create Sample Users"):
        try:
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
        st.markdown(f"The setup of `async_engine` and `AsyncSessionLocal` is central to Alex's async strategy. The `get_session` context manager ensures that database connections are properly acquired and released. The `pool_size` and `max_overflow` parameters are crucial for connection pooling, allowing the application to reuse existing connections and handle spikes in demand gracefully.")
    else:
        st.warning("Database not initialized. Please click the button above.")
```

#### Page: 3. Repository Pattern & N+1 Queries

**Content**:

```python
elif st.session_state.current_page == "3. Repository Pattern & N+1":
    st.title("3. Implementing the Repository Pattern and Solving N+1 Queries")

    st.markdown(f"To maintain a clean architecture and facilitate easier testing, Alex implements the Repository Pattern, abstracting database operations from the service layer. He also anticipates a common performance pitfall: the N+1 query problem, which arises when fetching a collection of parent objects and then, for each parent, executing a separate query to fetch its child objects. This can drastically degrade performance, especially when dealing with many related records. Alex addresses this with SQLAlchemy's eager loading techniques.")

    st.markdown(f"The Repository Pattern centralizes data access logic, making it easier to manage, test, and potentially swap out ORM or database technologies in the future. The N+1 query problem occurs when loading $N$ parent objects (e.g., users) and then subsequently executing $N$ additional queries to fetch their related child objects (e.g., scores), resulting in $N+1$ queries in total. This is inefficient. Eager loading techniques like `selectinload` reduce this to $1$ or $2$ queries, improving performance significantly.")

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
                                await asyncio.sleep(0.01) # Ensure distinct timestamps
                            st.success(f"3 sample scores added for user ID: {user_to_add_score_to}")
                    asyncio.run(add_scores())
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
                                    # Accessing `user.scores` here, if not eagerly loaded,
                                    # would typically trigger N additional queries (one per score)
                                    # due to SQLAlchemy's default lazy loading.
                                    # For this demo, we illustrate the *concept* of N+1.
                                    for score in user.scores: 
                                        scores_info.append(f"Score ID: {score.id}, AIR Score: {score.air_score}")
                                    st.write(f"**User (ID: {user.id}, Email: {user.email})**")
                                    st.write(f"**Scores (Simulated N+1):**")
                                    for s_info in scores_info:
                                        st.markdown(f"- {s_info}")
                                else:
                                    st.warning("User not found.")
                        asyncio.run(fetch_n1())
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
                        asyncio.run(fetch_eager())
                        end_time = time.time()
                        st.info(f"Time taken (Eager Loading): {end_time - start_time:.4f} seconds")
                    except Exception as e:
                        st.error(f"Error fetching with eager loading: {e}")
                else:
                    st.warning("Please enter a User ID.")

        st.markdown(f"By implementing `UserRepository`, Alex has created a clean boundary between the business logic and data access. The `get_user_with_scores_eager` method, utilizing `selectinload(User.scores)`, directly addresses the N+1 query problem by fetching related `AIRScore` objects in a minimal number of queries. This significantly reduces database load and improves response times, as observed by the difference in execution times.")

```

#### Page: 4. Caching with Redis

**Content**:

```python
elif st.session_state.current_page == "4. Caching with Redis":
    st.title("4. Optimizing Data Access with Redis Caching Strategies")

    st.markdown(f"InnovateAI's user profiles and their latest AI scores are frequently accessed, especially during the initial loading of the user dashboard. To offload the primary PostgreSQL database and accelerate response times for these hot data points, Alex decides to implement a caching layer using Redis. He needs to consider a read-through caching strategy, where data is fetched from the cache if available, otherwise from the database and then stored in the cache for subsequent requests.")

    st.markdown(f"Caching is critical for high-performance applications, reducing latency and database load by storing frequently accessed data in a fast, in-memory store like Redis. The read-through strategy is robust for frequently read, less frequently updated data. It simplifies cache management by encapsulating the cache-or-DB logic.")

    st.markdown(r"The effectiveness of caching is measured by the Cache Hit Ratio ($ H $): $$ H = \frac{{\text{{Number of Cache Hits}}}}{{\text{{Total Number of Requests}}}} $$")
    st.markdown(r"where $ H $ is the cache hit ratio. A higher $ H $ indicates better cache effectiveness. The Average Access Time ($ T_{{\text{avg}}} $) with caching is given by: $$ T_{{\text{avg}}} = H \times T_{{\text{cache}}} + (1-H) \times (T_{{\text{cache}}} + T_{{\text{database}}}) $$")
    st.markdown(r"where $ T_{{\text{cache}}} $ is cache access time and $ T_{{\text{database}}} $ is database access time. A good caching strategy aims to minimize $ T_{{\text{avg}}} $.")

    if not st.session_state.db_initialized:
        st.warning("Please initialize the database in the 'DB Connectivity & Pooling' section first.")
    else:
        st.subheader("Interactive Caching Demonstration")

        # Ensure a user exists for caching demo
        if not st.session_state.user_for_caching_id:
            try:
                async def create_cache_user_and_score():
                    async for session in get_db_session():
                        # Check if a specific user (e.g., Jane) already exists for demo continuity
                        user = await UserRepository(session).get_by_email("jane.doe@innovateai.com")
                        if not user: # Fallback if Jane isn't there
                            new_user = User(email="cache_demo@innovateai.com", name="Cache Demo User")
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
                        
                        st.info(f"User for caching demo (ID: `{st.session_state.user_for_caching_id}`) and latest score (ID: `{st.session_state.latest_airscore_id}`) ready.")
                asyncio.run(create_cache_user_and_score())
            except Exception as e:
                st.error(f"Error preparing user for caching demo: {e}")
        
        user_id_for_cache = st.session_state.user_for_caching_id

        if user_id_for_cache:
            st.markdown(f"Using **User ID**: `{user_id_for_cache}` for caching demonstration.")
            st.markdown(f"**Latest AIRScore ID**: `{st.session_state.latest_airscore_id}`")

            col1_cache, col2_cache = st.columns(2)
            with col1_cache:
                if st.button("Fetch User (Cached)", key="fetch_user_cached_btn"):
                    start_time = time.time()
                    try:
                        async def fetch_user_cached_interaction():
                            async for session in get_db_session():
                                cached_repo = CachedUserRepository(session, REDIS_CLIENT)
                                user = await cached_repo.get_by_id_cached(user_id_for_cache)
                                if user:
                                    st.write(f"**Cached User Details:**")
                                    st.json({
                                        "id": user.id, "email": user.email, "name": user.name,
                                        "occupation_code": user.occupation_code,
                                        "created_at": user.created_at.isoformat()
                                    })
                                else:
                                    st.warning("User not found in cache or DB.")
                        asyncio.run(fetch_user_cached_interaction())
                        end_time = time.time()
                        st.markdown(f"Time taken: {end_time - start_time:.4f} seconds")
                        # Simplified cache hit/miss tracking based on presence of REDIS_CLIENT.get call
                        if REDIS_CLIENT.get.called and REDIS_CLIENT.get.return_value is not None:
                             st.session_state.cache_hits += 1
                        else:
                             st.session_state.cache_misses += 1
                        REDIS_CLIENT.get.reset_mock() # Reset mock for next call if using mock client
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
                                if score:
                                    st.write(f"**Cached Latest AIRScore Details:**")
                                    st.json({
                                        "id": score.id, "user_id": score.user_id,
                                        "air_score": score.air_score, "occupation": score.occupation_code,
                                        "parameter_version": score.parameter_version,
                                        "created_at": score.created_at.isoformat()
                                    })
                                else:
                                    st.warning("Latest AIRScore not found in cache or DB.")
                        asyncio.run(fetch_score_cached_interaction())
                        end_time = time.time()
                        st.markdown(f"Time taken: {end_time - start_time:.4f} seconds")
                        if REDIS_CLIENT.get.called and REDIS_CLIENT.get.return_value is not None:
                             st.session_state.cache_hits += 1
                        else:
                             st.session_state.cache_misses += 1
                        REDIS_CLIENT.get.reset_mock()
                    except Exception as e:
                        st.error(f"Error fetching score with cache: {e}")
            
            st.markdown(f"---")
            st.subheader("Cache Invalidation")
            st.markdown(f"After an update, old cached data needs to be purged. Invalidate the cache for User ID: `{user_id_for_cache}`.")
            if st.button("Invalidate Cache for this User", key="invalidate_cache_btn"):
                try:
                    async def invalidate_cache_interaction():
                        # The session might not be strictly needed for cache invalidation for the redis client,
                        # but we pass it for consistency with repository pattern.
                        async for session in get_db_session(): 
                            cached_repo = CachedUserRepository(session, REDIS_CLIENT)
                            await cached_repo.invalidate_user_cache(user_id_for_cache)
                            st.success(f"Cache invalidated for user ID: {user_id_for_cache}")
                    asyncio.run(invalidate_cache_interaction())
                except Exception as e:
                    st.error(f"Error invalidating cache: {e}")

            st.markdown(f"---")
            st.subheader("Cache Metrics (Simulated)")
            total_requests = st.session_state.cache_hits + st.session_state.cache_misses
            hit_ratio = (st.session_state.cache_hits / total_requests) if total_requests > 0 else 0
            st.markdown(f"**Cache Hits:** {st.session_state.cache_hits}")
            st.markdown(f"**Cache Misses:** {st.session_state.cache_misses}")
            st.markdown(f"**Total Requests:** {total_requests}")
            st.markdown(f"**Cache Hit Ratio:** {hit_ratio:.2f}")

            st.markdown(f"Alex's implementation of `CachedUserRepository` successfully demonstrates a read-through caching strategy. The first request will typically result in a 'Cache MISS' (fetching from DB), and subsequent requests for the same data show a 'Cache HIT' (retrieving directly from Redis). The `invalidate_user_cache` method ensures data consistency by clearing stale cache entries after updates. This optimizes data access for read-heavy operations, improving the user experience.")

```

#### Page: 5. Eventing (Outbox Pattern)

**Content**:

```python
elif st.session_state.current_page == "5. Eventing (Outbox Pattern)":
    st.title("5. Building a Reliable Eventing System with the Outbox Pattern")

    st.markdown(f"Alex is tasked with ensuring that critical domain events (like an `AIRScore` being calculated or an `Assessment` completing) are reliably published to other microservices within InnovateAI, even in the face of temporary network issues or consumer downtime. He implements the Outbox Pattern, using the `DomainEvent` table as a robust buffer. This pattern is vital for ensuring that services remain loosely coupled and that event-driven architectures maintain data consistency.")

    st.markdown(f"In distributed systems, reliable communication between services is paramount. The Outbox Pattern ensures atomicity: a business operation and the recording of its corresponding domain event happen within a single database transaction. This guarantees that events are never lost if the publishing mechanism fails after the business operation succeeds but before the event is sent to the message broker. It leverages the ACID properties of the database.")

    st.markdown(r"The Outbox Pattern helps achieve **eventual consistency**. In a distributed system, data might not be immediately consistent across all services, but it will eventually converge. By guaranteeing events are eventually delivered, the Outbox pattern facilitates this convergence without requiring a complex two-phase commit protocol across services.")

    if not st.session_state.db_initialized:
        st.warning("Please initialize the database in the 'DB Connectivity & Pooling' section first.")
    else:
        st.subheader("Generate an AIRScore and a Pending Domain Event")

        # Ensure a user exists for eventing demo
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
                        st.info(f"User for eventing demo (ID: `{st.session_state.event_user_id}`) ready.")
                asyncio.run(create_event_user())
            except Exception as e:
                st.error(f"Error preparing user for eventing demo: {e}")

        user_id_for_event = st.session_state.event_user_id
        if user_id_for_event:
            st.markdown(f"Using **User ID**: `{user_id_for_event}` for eventing demonstration.")
            
            if st.button("Calculate & Store AIRScore (Creates Pending Event)", key="create_event_btn"):
                try:
                    async def calculate_score_and_event_interaction():
                        st.info("Calling `calculate_and_store_airscore`...")
                        score, event = await calculate_and_store_airscore(user_id_for_event, "AI_ASSESSOR")
                        st.success(f"AIRScore (ID: `{score.id}`) calculated and DomainEvent (ID: `{event.id}`, Type: `{event.event_type}`) recorded as 'pending'.")
                    
                    asyncio.run(calculate_score_and_event_interaction())
                except Exception as e:
                    st.error(f"Error calculating score and recording event: {e}")
            
            st.subheader("Event Publisher Status")
            col1_pub, col2_pub = st.columns(2)
            with col1_pub:
                if st.button("Start Event Publisher (Background)", key="start_publisher_btn", disabled=st.session_state.event_publisher_running):
                    st.session_state.event_publisher_running = True
                    st.session_state.publisher_stop_event.clear() # Ensure event is clear before starting

                    async def run_publisher_in_background():
                        # The `event_publisher` from source.py is a continuous loop.
                        # Running it directly with asyncio.run will block Streamlit.
                        # For a real Streamlit app, this would typically involve:
                        # 1. Using a separate thread/process for the publisher.
                        # 2. Or, for demo purposes, running a single iteration of the publisher logic
                        #    on a button click or `st.rerun` cycle.
                        # Given `source.py`'s `event_publisher` is a loop, we'll simulate it by
                        # running a few cycles within a non-blocking context, or rely on its
                        # `stop_condition` to manage its lifecycle.

                        # For this blueprint, we'll describe starting a task that would conceptually run in the background.
                        # The actual Streamlit implementation would need to manage this `asyncio.Task` carefully
                        # to prevent blocking the UI and to persist its state across reruns.
                        # A robust way is to create the task and store its reference in session state,
                        # checking its status on subsequent reruns.
                        
                        # Here, for the blueprint, we will define a simple loop that runs a few iterations
                        # before stopping, to show the effect. A truly continuous publisher needs more advanced Streamlit techniques.
                        
                        st.info("Event publisher starting... Processing up to 3 cycles of events.")
                        processed_count = 0
                        for _ in range(3): # Simulate 3 polling cycles
                            if st.session_state.publisher_stop_event.is_set():
                                break
                            async for session in get_db_session(): # Publisher needs a session for each poll
                                event_repo = DomainEventRepository(session)
                                pending_events = await event_repo.get_pending_events(limit=5)
                                if pending_events:
                                    event_ids_to_publish = [event.id for event in pending_events]
                                    await event_repo.mark_as_published(event_ids_to_publish)
                                    processed_count += len(event_ids_to_publish)
                                    st.markdown(f"*(Publisher activity)* Processed {len(event_ids_to_publish)} events.")
                                # Simulate sleep between polls (non-blocking for this wrapper)
                                await asyncio.sleep(0.5) 
                        st.session_state.event_publisher_running = False # Mark as stopped after simulated cycles
                        st.success(f"Event publisher finished its simulated run, processed {processed_count} events.")
                        # After simulated run, refresh display
                        asyncio.run(refresh_events_status_internal())


                    # Start the background publisher. Using asyncio.create_task and a non-blocking `asyncio.sleep`
                    # is key. For a Streamlit app to *truly* run something in the background
                    # that persists across reruns, it's more complex (e.g., using a separate thread/process or
                    # Streamlit's new `st.experimental_rerun` with careful task management).
                    # Here, we'll just run a limited, simulated set of cycles to show the effect.
                    asyncio.create_task(run_publisher_in_background())
                    st.experimental_rerun() # Rerun to update button states and display immediately.
            
            with col2_pub:
                if st.button("Stop Event Publisher", key="stop_publisher_btn", disabled=not st.session_state.event_publisher_running):
                    st.session_state.publisher_stop_event.set() # Signal the publisher to stop
                    st.session_state.event_publisher_running = False
                    st.warning("Event publisher signalled to stop (will stop after current cycle).")
                    st.experimental_rerun() # Rerun to update button states.

            st.subheader("Current Event Status")
            
            async def refresh_events_status_internal():
                async for session in get_db_session():
                    event_repo = DomainEventRepository(session)
                    pending_events = await event_repo.get_pending_events(limit=10)
                    all_events_result = await session.execute(select(DomainEvent).order_by(DomainEvent.created_at.desc()))
                    all_events = all_events_result.scalars().all()
                    
                    st.session_state.pending_events_display = [
                        {"ID": e.id, "Type": e.event_type, "Status": e.status, "Created": e.created_at.strftime("%Y-%m-%d %H:%M:%S")}
                        for e in pending_events
                    ]
                    st.session_state.processed_events_display = [
                        {"ID": e.id, "Type": e.event_type, "Status": e.status, "Published": e.published_at.strftime("%Y-%m-%d %H:%M:%S") if e.published_at else "N/A"}
                        for e in all_events if e.status == "published"
                    ]

            # Button to manually refresh event status
            if st.button("Refresh Event Status Now", key="refresh_event_status_btn"):
                asyncio.run(refresh_events_status_internal())
            
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
```
