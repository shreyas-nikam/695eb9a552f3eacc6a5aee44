Here's a comprehensive `README.md` file for your Streamlit application lab project, designed for clarity, professionalism, and ease of use.

---

# QuLab: Data Architecture & Persistence Lab Project 🚀

## Project Title: Scaling the AI Backend's Data Layer

This Streamlit application serves as an interactive lab project for exploring advanced data architecture and persistence patterns in a Python backend, specifically tailored for an AI-powered assessment platform. It demonstrates key concepts using SQLAlchemy 2.0 with asynchronous drivers, Redis for caching, and the Outbox Pattern for reliable eventing.

The project is designed to simulate the challenges faced by a Senior Software Engineer at InnovateAI Solutions in building a robust, performant, and reliable data layer for an AI assessment platform.

## Features ✨

This lab project interactively showcases the following core features and architectural patterns:

*   **1. Data Models with SQLAlchemy 2.0**:
    *   Defines `User`, `Assessment`, `AIRScore`, and `DomainEvent` models using SQLAlchemy's modern declarative mapping and type hints.
    *   Illustrates one-to-many relationships and cascade operations.
    *   Interactive demonstration for creating new user records.

*   **2. Asynchronous DB Connectivity & Connection Pooling**:
    *   Establishes asynchronous database connections using `sqlite+aiosqlite` (configured for local SQLite for demonstration, easily swappable for `asyncpg` with PostgreSQL in production).
    *   Showcases connection pooling setup for efficient resource management.
    *   Includes a setup step to initialize the database schema and populate sample users.

*   **3. Repository Pattern & N+1 Query Problem Resolution**:
    *   Implements a `UserRepository` to abstract data access logic, promoting a clean architecture.
    *   Demonstrates and mitigates the N+1 query problem using SQLAlchemy's eager loading techniques (`selectinload`) to optimize fetching related data.
    *   Allows adding sample scores to users to simulate data for the N+1 problem.

*   **4. Caching with Redis**:
    *   Integrates Redis as a caching layer using a `CachedUserRepository`.
    *   Implements a read-through caching strategy for frequently accessed user profiles and latest AI scores.
    *   Provides functionality for explicit cache invalidation.
    *   Simulates and tracks cache hits and misses to demonstrate caching effectiveness.

*   **5. Reliable Eventing (Outbox Pattern)**:
    *   Utilizes a `DomainEvent` table to implement the Outbox Pattern for reliable publishing of critical events (e.g., `AIRScoreCalculated`).
    *   Ensures atomicity: business operation and event recording occur within a single database transaction.
    *   Simulates a background event publisher that polls for pending events, processes them, and marks them as published.

*   **Streamlit Interactive UI**: All features are presented through an intuitive Streamlit interface, allowing users to interact with and observe the architectural patterns in real-time.

## Getting Started 🚀

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python**: Version 3.9 or higher.
*   **Redis Server**: For the caching and eventing sections to function fully, a local Redis server instance is required.
    *   [Install Redis](https://redis.io/docs/getting-started/installation/) (e.g., via Docker, Homebrew, or official packages).
    *   The application assumes Redis is running on `redis://localhost:6379/0`. If Redis is not available, the application will gracefully fall back to a mock Redis client for basic functionality (though caching/eventing won't be demonstrated effectively).

### Installation

1.  **Clone the repository** (or download the `app.py` file if this is a single-file distribution):

    ```bash
    git clone https://github.com/your-username/quolab-data-architecture.git
    cd quolab-data-architecture
    ```

2.  **Create a virtual environment** and activate it (recommended):

    ```bash
    python -m venv venv
    # On Windows:
    .\venv\Scripts\activate
    # On macOS/Linux:
    source venv/bin/activate
    ```

3.  **Install the required Python packages**:
    Create a `requirements.txt` file with the following content:

    ```
    streamlit
    sqlalchemy
    aiosqlite
    redis[async]
    ```

    Then install:

    ```bash
    pip install -r requirements.txt
    ```

## Usage 👨‍💻

To run the Streamlit application:

1.  **Ensure your virtual environment is activated.**
2.  **Navigate to the project directory** (where `app.py` is located).
3.  **Run the Streamlit application**:

    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser.

### Interacting with the Application

The Streamlit app is divided into several sections, accessible via the **sidebar navigation**:

1.  **Introduction**: Overview of the lab and its objectives.
2.  **1. Data Models**:
    *   Observe the SQLAlchemy model definitions.
    *   Use the form to **create a new user** and see it persisted in the (in-memory) database.
3.  **2. DB Connectivity & Pooling**:
    *   Click "Initialize In-Memory SQLite Database & Create Sample Users" to set up the database and populate initial data. This is a prerequisite for all other sections.
4.  **3. Repository Pattern & N+1**:
    *   Add sample scores for existing users.
    *   Compare the performance of "Simulated N+1" queries vs. "Eager Loading" using the provided buttons and observe the time differences.
5.  **4. Caching with Redis**:
    *   Fetch user and score data using the cached repository.
    *   Observe the "cache hits" and "cache misses" (simulated).
    *   Test cache invalidation.
    *   *Requires a running Redis server for full demonstration.*
6.  **5. Eventing (Outbox Pattern)**:
    *   Generate a new AIRScore, which automatically creates a "pending" `DomainEvent`.
    *   Start the simulated "Event Publisher" to process pending events and mark them as "published".
    *   Observe the transitions of event statuses.

## Project Structure 📁

This project is contained within a single `app.py` file for simplicity in a lab setting. However, it's logically divided into distinct sections:

```
.
├── app.py                  # Main Streamlit application file
└── requirements.txt        # Python dependencies
```

Within `app.py`, the code is structured as follows:

*   **Configuration**: Database URL, Redis setup.
*   **SQLAlchemy Base and Mixins**: `Base`, `TimestampMixin`.
*   **Models**: `User`, `Assessment`, `AIRScore`, `DomainEvent`.
*   **Database Engine and Session Setup**: Async engine, session factory, `get_session` context manager.
*   **Repositories**: `BaseRepository`, `UserRepository`, `AIRScoreRepository`, `DomainEventRepository`.
*   **Caching with Redis**: `CachedUserRepository` (inherits from `UserRepository`).
*   **Helper Functions**: `init_db`, `run_db_setup_and_create_user`, `create_sample_airscore`, `calculate_and_store_airscore`.
*   **Streamlit UI**: `st.set_page_config`, sidebar navigation, and content for each lab section.

## Technology Stack 🛠️

*   **Python**: The core programming language.
*   **Streamlit**: For creating interactive web applications with pure Python.
*   **SQLAlchemy 2.0**: Powerful SQL toolkit and Object-Relational Mapper (ORM) for Python, used for asynchronous database interactions.
*   **aiosqlite**: Asynchronous driver for SQLite, used with SQLAlchemy. (In a production scenario, `asyncpg` would be used for PostgreSQL).
*   **Redis**: An in-memory data store, used for caching and as a foundation for the Outbox Pattern's event processing.
*   **asyncio**: Python's built-in library for writing concurrent code using the `async/await` syntax.

## Contributing 🤝

This is a lab project, primarily for learning and demonstration. However, if you find issues or have suggestions for improvements, feel free to open an issue or submit a pull request.

## License 📄

This project is licensed under the MIT License - see the `LICENSE` file (if included) for details.

## Contact 📧

For any questions or feedback, please reach out to:

*   **Your Name/Org**: QuLab Team
*   **Email**: support@quantuniversity.com
*   **Website**: [QuantUniversity](https://www.quantuniversity.com)

---