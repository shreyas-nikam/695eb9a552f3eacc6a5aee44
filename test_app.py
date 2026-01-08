
import pytest
from streamlit.testing.v1 import AppTest
import asyncio
from unittest.mock import patch, AsyncMock

# Helper to initialize DB and create sample users for tests that require it.
# This function interacts with the Streamlit app's UI to trigger DB initialization.
async def _initialize_db_and_users(at: AppTest):
    """
    Navigates to the '2. DB Connectivity & Pooling' page and clicks the
    'Initialize In-Memory SQLite Database & Create Sample Users' button.
    Verifies the success messages and updates to session state.
    """
    # Navigate to "2. DB Connectivity & Pooling"
    at.sidebar.selectbox[0].set_value("2. DB Connectivity & Pooling").run()
    
    # Click the initialization button (assuming it's the first button on this page)
    at.button[0].click().run()
    
    # Assert success messages and session state updates
    assert at.success[0].value.startswith("Initialized DB and created user:")
    assert at.success[1].value.startswith("Created user:")
    assert at.session_state["db_initialized"] is True
    assert at.session_state["user_alex_id"] is not None
    assert at.session_state["user_jane_id"] is not None
    
    return at.session_state["user_alex_id"], at.session_state["user_jane_id"]

def test_initial_page_and_redis_status():
    """
    Verifies the application starts on the Introduction page and displays
    the correct title and Redis connection status.
    """
    at = AppTest.from_file("app.py").run()
    
    # Check if the main title and header of the introduction page are present
    assert at.title[0].value == "QuLab: Data Architecture & Persistence"
    assert at.header[0].value == "Introduction: Scaling the AI Backend's Data Layer"
    
    # Verify Redis status in the sidebar. The app mocks Redis if unavailable.
    assert "Redis Status:" in at.sidebar.markdown[1].value
    assert ("Using Mock Redis Client" in at.sidebar.markdown[1].value or
            "Connected to local Redis." in at.sidebar.markdown[1].value)

def test_sidebar_navigation():
    """
    Ensures that selecting different options in the sidebar correctly changes
    the current page and updates the displayed content.
    """
    at = AppTest.from_file("app.py").run()

    # Test navigation to "1. Data Models"
    at.sidebar.selectbox[0].set_value("1. Data Models").run()
    assert at.session_state["current_page"] == "1. Data Models"
    assert at.title[0].value == "1. Defining the Core Data Schema with SQLAlchemy 2.0"

    # Test navigation to "3. Repository Pattern & N+1"
    at.sidebar.selectbox[0].set_value("3. Repository Pattern & N+1").run()
    assert at.session_state["current_page"] == "3. Repository Pattern & N+1"
    assert at.title[0].value == "3. Implementing the Repository Pattern and Solving N+1 Queries"

def test_db_connectivity_and_pooling_initialization():
    """
    Tests the database initialization process on the '2. DB Connectivity & Pooling' page,
    verifying initial warnings and successful setup.
    """
    at = AppTest.from_file("app.py").run()

    # Navigate to "2. DB Connectivity & Pooling"
    at.sidebar.selectbox[0].set_value("2. DB Connectivity & Pooling").run()

    # Verify that a warning is displayed when the database is not yet initialized
    assert at.warning[0].value == "Database not initialized. Please click the button above."

    # Perform database initialization using the helper function
    alex_id, jane_id = asyncio.run(_initialize_db_and_users(at))

    # Assert that success messages are displayed and session state reflects initialization
    assert at.success[2].value == "Database is initialized and ready!"
    assert f"**Alex Smith User ID:** `{alex_id}`" in at.markdown[2].value
    assert f"**Jane Doe User ID:** `{jane_id}`" in at.markdown[3].value

def test_data_models_create_user():
    """
    Tests the "Create New User" form on the "1. Data Models" page,
    including successful user creation and handling of duplicate emails.
    """
    at = AppTest.from_file("app.py").run()

    # Initialize the database first, as the "1. Data Models" page's interactive
    # section depends on `st.session_state.db_initialized` being True.
    asyncio.run(_initialize_db_and_users(at))

    # Navigate to "1. Data Models"
    at.sidebar.selectbox[0].set_value("1. Data Models").run()

    # Test creating a new user successfully
    at.text_input(key="new_user_email_input").set_value("test_user_new@example.com").run()
    at.text_input(key="new_user_name_input").set_value("Test User New").run()
    at.form_submit_button("Create User").click().run()

    assert at.success[0].value.startswith("User 'Test User New' created with ID:")
    assert at.session_state["created_user_id"] is not None

    # Test creating a user with an existing email (should trigger an IntegrityError
    # in the backend and display the corresponding error message in the app).
    at.text_input(key="new_user_email_input").set_value("test_user_new@example.com").run()
    at.text_input(key="new_user_name_input").set_value("Another User").run()
    at.form_submit_button("Create User").click().run()
    assert at.error[0].value == "User with email 'test_user_new@example.com' already exists. Please use a unique email."

def test_repository_pattern_n_plus_1():
    """
    Tests the functionality on the "3. Repository Pattern & N+1" page,
    including adding sample scores and demonstrating N+1 vs. eager loading.
    """
    at = AppTest.from_file("app.py").run()

    # Initialize the database to ensure users exist and the app is ready.
    alex_id, jane_id = asyncio.run(_initialize_db_and_users(at))

    # Navigate to "3. Repository Pattern & N+1"
    at.sidebar.selectbox[0].set_value("3. Repository Pattern & N+1").run()

    # Add 3 sample scores for Alex to demonstrate N+1
    at.text_input(key="n1_user_id_input").set_value(alex_id).run()
    at.button[0].click().run() # "Add 3 Sample Scores for User" button
    assert at.success[0].value == f"3 sample scores added for user ID: {alex_id}"

    # Fetch User & Scores using the "Simulated N+1" approach
    at.text_input(key="fetch_user_id_n1").set_value(alex_id).run()
    at.button(key="n1_button").click().run()
    assert f"**User (ID: {alex_id}, Email: alex.smith@innovateai.com)**" in at.write[0].value
    assert "**Scores (Simulated N+1):**" in at.write[1].value
    # Check for the presence of markdown elements representing the fetched scores
    assert len(at.markdown) >= 3 # Expecting at least 3 score display elements
    assert "- Score ID:" in at.markdown[0].value # Check content of a score item

    # Fetch User & Scores using "Eager Loading"
    at.button(key="eager_button").click().run()
    assert f"**User (ID: {alex_id}, Email: alex.smith@innovateai.com)**" in at.write[2].value
    assert "**Scores (Eager Loaded):**" in at.write[3].value
    # Expecting additional markdown elements for eager loaded scores (after the N+1 ones)
    assert len(at.markdown) >= 6 
    assert "- Score ID:" in at.markdown[3].value # Check content of an eager loaded score item
    assert at.session_state["retrieved_user_with_scores"] is not None

# Patch `source.REDIS_CLIENT` to use a mock for controlled testing of caching logic.
@patch("source.REDIS_CLIENT", new_callable=AsyncMock)
def test_caching_with_redis(mock_redis_client):
    """
    Tests the caching functionality on the "4. Caching with Redis" page,
    simulating cache hits/misses and invalidation using a mocked Redis client.
    """
    at = AppTest.from_file("app.py").run()

    # Initialize the database for user and score data.
    asyncio.run(_initialize_db_and_users(at))

    # Navigate to "4. Caching with Redis"
    at.sidebar.selectbox[0].set_value("4. Caching with Redis").run()
    
    # Run once more to allow the async setup for the caching demo user
    # (create_cache_user_and_score) to complete and populate session state.
    at.run() 
    assert at.session_state["user_for_caching_id"] is not None
    assert at.session_state["latest_airscore_id"] is not None
    cache_user_id = at.session_state["user_for_caching_id"]
    
    # Reset cache metrics in session state for a clean test run of the caching logic.
    at.session_state["cache_hits"] = 0
    at.session_state["cache_misses"] = 0

    # Simulate a Cache Miss when fetching a user
    mock_redis_client.get.return_value = None # No item found in the mock cache
    at.button(key="fetch_user_cached_btn").click().run()
    assert at.write[0].value == "**Cached User Details:**"
    assert at.json[0].get_element_by_key("id").value == cache_user_id # Data fetched from DB
    assert at.session_state["cache_misses"] == 1
    assert at.session_state["cache_hits"] == 0
    mock_redis_client.get.assert_called_once()
    mock_redis_client.get.reset_mock() # Clear mock call history for the next assertion

    # Simulate a Cache Hit when fetching the same user
    mock_redis_client.get.return_value = '{"id": "mock_user_id", "email": "mock@innovateai.com", "name": "Mock User", "occupation_code": "MOCK", "created_at": "2023-01-01T00:00:00+00:00"}'
    at.button(key="fetch_user_cached_btn").click().run()
    assert at.write[0].value == "**Cached User Details:**"
    assert at.json[0].get_element_by_key("id").value == "mock_user_id" # Data fetched from mock cache
    assert at.session_state["cache_misses"] == 1 # Miss count remains unchanged for this interaction
    assert at.session_state["cache_hits"] == 1 # Hit count increments
    mock_redis_client.get.assert_called_once()
    mock_redis_client.get.reset_mock()

    # Simulate a Cache Miss when fetching the latest AIRScore
    mock_redis_client.get.return_value = None
    at.button(key="fetch_score_cached_btn").click().run()
    assert at.write[1].value == "**Cached Latest AIRScore Details:**"
    assert at.json[0].get_element_by_key("user_id").value == cache_user_id # Data fetched from DB
    assert at.session_state["cache_misses"] == 2
    assert at.session_state["cache_hits"] == 1
    mock_redis_client.get.assert_called_once()
    mock_redis_client.get.reset_mock()

    # Simulate a Cache Hit when fetching the same AIRScore
    mock_redis_client.get.return_value = '{"id": "mock_airscore_id", "user_id": "mock_user_id", "air_score": 99.9, "occupation": "MOCK_OCC", "parameter_version": "v1", "created_at": "2023-01-01T00:00:00+00:00"}'
    at.button(key="fetch_score_cached_btn").click().run()
    assert at.write[1].value == "**Cached Latest AIRScore Details:**"
    assert at.json[0].get_element_by_key("id").value == "mock_airscore_id" # Data fetched from mock cache
    assert at.session_state["cache_misses"] == 2
    assert at.session_state["cache_hits"] == 2
    mock_redis_client.get.assert_called_once()
    mock_redis_client.get.reset_mock()
    
    # Test Cache Invalidation
    mock_redis_client.delete.return_value = 1 # Simulate successful deletion
    at.button(key="invalidate_cache_btn").click().run()
    assert at.success[0].value == f"Cache invalidated for user ID: {cache_user_id}"
    mock_redis_client.delete.assert_called_once()
    # Verify the correct cache key was used for deletion
    mock_redis_client.delete.assert_called_with(f"user:{cache_user_id}")
    mock_redis_client.delete.reset_mock()


def test_eventing_outbox_pattern():
    """
    Tests the Outbox Pattern implementation on the "5. Eventing (Outbox Pattern)" page,
    including creating events, starting a simulated publisher, and verifying event status.
    """
    at = AppTest.from_file("app.py").run()

    # Initialize the database to support event creation.
    asyncio.run(_initialize_db_and_users(at))

    # Navigate to "5. Eventing (Outbox Pattern)"
    at.sidebar.selectbox[0].set_value("5. Eventing (Outbox Pattern)").run()
    
    # Run once more to ensure the async setup for the eventing demo user
    # (create_event_user) is complete and reflected in session state.
    at.run()
    assert at.session_state["event_user_id"] is not None
    event_user_id = at.session_state["event_user_id"]

    # Generate an AIRScore, which should also create a pending DomainEvent
    at.button(key="create_event_btn").click().run()
    assert at.success[0].value.startswith("AIRScore (ID:")
    assert "recorded as 'pending'." in at.success[0].value

    # Refresh Event Status to confirm the pending event is visible
    at.button(key="refresh_event_status_btn").click().run()
    assert len(at.session_state["pending_events_display"]) > 0
    assert at.session_state["pending_events_display"][0]["Status"] == "pending"

    # Start the simulated Event Publisher (it runs for 3 cycles and stops).
    # `asyncio.run` in the app means this button click will block until
    # the publisher's simulated run is complete.
    at.button(key="start_publisher_btn").click().run()
    # After completion, a success message should be displayed.
    assert "Event publisher finished its simulated run" in at.success[1].value
    assert at.session_state["event_publisher_running"] is False

    # Refresh Event Status again to confirm the event has been processed (published)
    at.button(key="refresh_event_status_btn").click().run()
    
    assert len(at.session_state["pending_events_display"]) == 0 # No pending events after processing
    assert len(at.session_state["processed_events_display"]) > 0
    assert at.session_state["processed_events_display"][0]["Status"] == "published"

    # Verify that the "Stop Event Publisher" button is disabled since the publisher is not running.
    assert at.button(key="stop_publisher_btn").disabled is True
