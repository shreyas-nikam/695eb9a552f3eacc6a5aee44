# QuLab: Data Architecture & Persistence Lab

![QuantUniversity Logo](https://www.quantuniversity.com/assets/img/logo5.jpg)

## Project Title and Description

**QuLab: Data Architecture & Persistence for InnovateAI Solutions**

This repository contains a Streamlit application serving as a hands-on lab project focusing on critical data architecture and persistence patterns for a scalable AI backend. It simulates the workflow of Alex, a Senior Software Engineer at InnovateAI Solutions, as he tackles challenges in designing and implementing a robust, performant, and reliable data layer for an AI-powered assessment platform.

The lab demonstrates practical applications of modern data persistence using SQLAlchemy 2.0, connection pooling, the Repository Pattern, Redis for caching, and an eventing system built on the Outbox Pattern. It's designed to help developers understand and apply advanced database concepts in asynchronous Python applications.

## Features

This interactive Streamlit application covers the following key data architecture and persistence concepts:

1.  **Data Models with SQLAlchemy 2.0**:
    *   Defines core data schemas (`User`, `Assessment`, `AIRScore`, `DomainEvent`) using SQLAlchemy 2.0's declarative mapping and type hints.
    *   Demonstrates object-relational mapping (ORM) with relationships (one-to-many).
    *   Interactive form to create new `User` records.

2.  **Asynchronous DB Connectivity & Connection Pooling**:
    *   Illustrates the setup of asynchronous database connections using SQLAlchemy 2.0 with the `asyncpg` driver (conceptually, for PostgreSQL) and an in-memory SQLite for the demo.
    *   Explains connection pooling for efficient resource management and maximizing throughput in I/O-bound applications.
    *   **Interactive**: One-click initialization of the database schema and creation of sample users.

3.  **Repository Pattern & N+1 Queries**:
    *   Implements the Repository Pattern for abstracting data access logic, promoting clean architecture and testability.
    *   Demonstrates the N+1 query problem and its solution using eager loading techniques (`selectinload`) to optimize data retrieval.
    *   **Interactive**: Compare "Simulated N+1" vs. "Eager Loading" performance for fetching user data with associated scores.

4.  **Caching with Redis**:
    *   Integrates Redis as a caching layer to offload the primary database for frequently accessed data (e.g., user profiles, latest AI scores).
    *   Employs a read-through caching strategy.
    *   **Interactive**: Fetch user and score data with and without cache, observe simulated cache hits/misses, and invalidate cache entries. Gracefully handles Redis unavailability by falling back to a mock client.

5.  **Reliable Eventing (Outbox Pattern)**:
    *   Implements the Outbox Pattern to guarantee atomicity between a business operation (e.g., calculating an AIRScore) and the recording of its corresponding domain event within a single database transaction.
    *   Simulates an event publisher that processes pending domain events.
    *   **Interactive**: Generate an AIRScore (which creates a pending event), run a simulated event publisher, and observe events transitioning from 'pending' to 'published'.

## Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

*   **Python 3.8+**: Ensure Python is installed on your system.
*   **pip**: Python package installer (usually comes with Python).
*   **Redis (Optional but Recommended)**: For the full caching and eventing experience, have a Redis server running locally (default `redis://localhost:6379`). If Redis is not available, the application will gracefully fall back to a mock Redis client.

### Installation

1.  **Clone the repository**:

    ```bash
    git clone https://github.com/your-username/quolab-data-architecture.git
    cd quolab-data-architecture
    ```

2.  **Install Python dependencies**:
    It's recommended to use a virtual environment.

    ```bash
    python -m venv venv
    source venv/bin/activate   # On Windows: venv\Scripts\activate
    pip install -r requirements.txt
    ```

    _Note: The `requirements.txt` would typically contain:_

    ```
    streamlit
    sqlalchemy
    asyncpg  # Required for async SQLAlchemy with PostgreSQL, but also useful for local setup via its dependencies.
    redis    # For redis.asyncio
    ```
    *(For this specific lab, the `source.py` is assumed to contain database models and helper functions, and `app.py` is the primary Streamlit runner.)*

## Usage

1.  **Ensure Redis is running (Optional)**:
    If you wish to use a real Redis instance for the caching and eventing sections, start your Redis server.

    ```bash
    # Example for starting Redis via Docker
    docker run -d --name my-redis -p 6379:6379 redis
    ```
    If Redis is not running, the application will seamlessly use a mock client, though the performance benefits of caching won't be demonstrated with real speed.

2.  **Run the Streamlit application**:

    ```bash
    streamlit run app.py
    ```

    This will open the application in your default web browser.

3.  **Navigate the Lab**:
    *   Use the sidebar on the left to navigate through the different sections of the lab project.
    *   **Start with "2. DB Connectivity & Pooling"** to initialize the in-memory SQLite database and create sample users. This is a prerequisite for interacting with subsequent sections.
    *   Follow the instructions and interact with the buttons and input fields in each section to explore the concepts.

## Project Structure

*   `app.py`: The main Streamlit application script. This file orchestrates the UI, handles user interactions, and integrates with the backend logic.
*   `source.py`: (Assumed) This file contains the core business logic, including SQLAlchemy model definitions, database session management, repository implementations, caching logic, and eventing mechanisms. It defines asynchronous functions that are called from `app.py`.
*   `requirements.txt`: Lists all Python dependencies required to run the application.

## Technology Stack

*   **Frontend/UI**:
    *   [Streamlit](https://streamlit.io/) - For rapidly building interactive web applications in Python.
*   **Backend/Data Layer**:
    *   [Python 3.8+](https://www.python.org/) - The primary programming language.
    *   [SQLAlchemy 2.0](https://www.sqlalchemy.org/) - SQL Toolkit and Object Relational Mapper for Python (with async support).
    *   [Asyncpg](https://github.com/MagicStack/asyncpg) - A fast PostgreSQL database driver for Python/asyncio (conceptually used, even if demo uses SQLite).
    *   [Redis](https://redis.io/) and [redis.asyncio](https://redis-py.readthedocs.io/en/stable/asyncio.html) - For caching and pub/sub capabilities.
    *   [SQLite](https://www.sqlite.org/index.html) - Used for the in-memory database demonstration due to its lightweight nature.
    *   [Alembic](https://alembic.sqlalchemy.org/) - Database migration tool (mentioned conceptually, actual migrations not included in `app.py`).
    *   [Asyncio](https://docs.python.org/3/library/asyncio.html) - Python's built-in framework for writing concurrent code using the async/await syntax.

## Contributing

This is a lab project, primarily for learning. However, if you find any bugs or have suggestions for improvements, feel free to open an issue or submit a pull request.

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
*(A `LICENSE` file should be created in the root of the repository if this is a real project.)*

## Contact

For any questions or further information, please contact:

*   **QuantUniversity** - [quantuniversity.com](https://www.quantuniversity.com/)
*   **Email**: info@quantuniversity.com
*   **Project Link**: [https://github.com/your-username/quolab-data-architecture](https://github.com/your-username/quolab-data-architecture) (Replace with actual link)

---
**Disclaimer**: The `source.py` file is assumed to be structured correctly with asynchronous functions designed to be called via `asyncio.run()` in `app.py`. The provided `app.py` code correctly demonstrates this interaction pattern.
