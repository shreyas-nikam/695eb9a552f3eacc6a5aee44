
from streamlit.testing.v1 import AppTest
from unittest import mock
import asyncio
import uuid
import pytest
from datetime import datetime

# It's good practice to import source and patch get_session_patchable here
# as the app.py relies on it being patched before its own execution.
import source
from source import *

@pytest.fixture(autouse=True)
def patch_get_session_for_tests():
    """
    Fixture to ensure source.get_session_patchable is correctly set
    for all tests that might interact with the database.
    This mimics the app's own patching logic.
    """
    source.get_session_patchable = get_db_session
    yield
    # No teardown needed as AppTest creates a fresh environment per run

@pytest.fixture
def at():
    """
    Fixture to load the app and run it initially.
    This also handles the implicit mock Redis setup as per app.py's logic.
    """
    # AppTest will run the app code, which attempts to connect to Redis.
    # If it fails (which it will in most test environments without a running Redis),
    # the app's logic will automatically set up a mock Redis client.
    app_test_instance = AppTest.from_file("app.py")
    app_test_instance.run()
    return app_test_instance

def test_initial_load_and_introduction_page(at):
    """
    Test that the app loads correctly and displays the Introduction page content.
    """
    assert at.title[0].value == "QuLab: Data Architecture & Persistence"
    assert at.sidebar.selectbox[0].value == "Introduction"
    assert "Redis Status: Using Mock Redis (Real Redis unavailable)." in at.sidebar.markdown[0].value
    assert at.header[0].value == "Introduction: Scaling the AI Backend's Data Layer"
    assert "Alex, Senior Software Engineer at InnovateAI Solutions." in at.markdown[1].value

def test_navigation(at):
    """
    Test that navigating through the sidebar updates the page content.
    """
    at.sidebar.selectbox[0].set_value("1. Data Models").run()
    assert at.title[0].value == "1. Defining the Core Data Schema with SQLAlchemy 2.0"
    assert at.sidebar.selectbox[0].value == "1. Data Models"

    at.sidebar.selectbox[0].set_value("2. DB Connectivity & Pooling").run()
    assert at.title[0].value == "2. Establishing Asynchronous Database Connectivity and Connection Pooling"
    assert at.sidebar.selectbox[0].value == "2. DB Connectivity & Pooling"

def test_db_initialization_and_sample_users(at):
    """
    Test the database initialization and sample user creation.
    """
    # Navigate to DB Connectivity page
    at.sidebar.selectbox[0].set_value("2. DB Connectivity & Pooling").run()
    assert not at.session_state["db_initialized"]
    assert "Database not initialized. Please click the button above." in at.warning[0].value

    # Click the initialize button
    at.button[0].click().run()

    assert at.session_state["db_initialized"]
    assert at.session_state["user_alex_id"] is not None
    assert at.session_state["user_jane_id"] is not None
    assert "Initialized DB and created user: 'Alex Smith'" in at.success[0].value
    assert "Created user: 'Jane Doe'" in at.success[1].value
    assert "Database is initialized and ready!" in at.success[2].value
    assert f"Alex Smith User ID: `{at.session_state['user_alex_id']}`" in at.markdown[1].value

def test_create_new_user_data_models_page(at):
    """
    Test creating a new user on the 'Data Models' page.
    Requires DB to be initialized first.
    """
    # Initialize DB first
    at.sidebar.selectbox[0].set_value("2. DB Connectivity & Pooling").run()
    at.button[0].click().run()
    assert at.session_state["db_initialized"]

    # Navigate to Data Models page
    at.sidebar.selectbox[0].set_value("1. Data Models").run()

    # Test with missing email
    at.text_input[1].set_value("Test User").run() # New User Name
    at.form[0].submit().run()
    assert "Please provide both email and name for the new user." in at.warning[0].value

    # Test with valid data
    email = f"new_user_{uuid.uuid4()}@example.com"
    at.text_input[0].set_value(email).run() # New User Email
    at.text_input[1].set_value("Test User").run() # New User Name
    at.form[0].submit().run()

    assert at.session_state["created_user_id"] is not None
    assert f"User 'Test User' created with ID: {at.session_state['created_user_id']}" in at.success[0].value
    assert f"Last Created User ID: `{at.session_state['created_user_id']}`" in at.markdown[4].value

    # Test duplicate email
    at.text_input[0].set_value(email).run() # New User Email
    at.text_input[1].set_value("Another User").run()
    at.form[0].submit().run()
    assert f"User with email '{email}' already exists. Please use a unique email." in at.error[0].value

def test_nplus1_and_eager_loading(at):
    """
    Test adding scores and demonstrating N+1 vs Eager Loading.
    Requires DB to be initialized first.
    """
    # Initialize DB first
    at.sidebar.selectbox[0].set_value("2. DB Connectivity & Pooling").run()
    at.button[0].click().run()
    assert at.session_state["db_initialized"]

    # Navigate to Repository Pattern & N+1 page
    at.sidebar.selectbox[0].set_value("3. Repository Pattern & N+1").run()

    user_id = at.session_state["user_alex_id"]
    at.text_input[0].set_value(user_id).run() # User ID to add scores for

    # Add sample scores
    at.button[0].click().run()
    assert f"3 sample scores added for user ID: {user_id}" in at.success[0].value

    # Fetch user & scores (Simulated N+1)
    at.text_input[1].set_value(user_id).run() # User ID to fetch
    at.button[1].click().run() # Fetch User & Scores (Simulated N+1)
    assert f"User (ID: {user_id}" in at.write[0].value
    assert "Scores (Simulated N+1):" in at.write[1].value
    assert "Time taken (Simulated N+1):" in at.info[0].value
    assert "Score ID:" in at.markdown[4].value # Check for score info

    # Fetch user & scores (Eager Loading)
    at.button[2].click().run() # Fetch User & Scores (Eager Loading)
    assert f"User (ID: {user_id}" in at.write[2].value
    assert "Scores (Eager Loaded):" in at.write[3].value
    assert "Time taken (Eager Loading):" in at.info[1].value
    assert "Score ID:" in at.markdown[5].value # Check for score info
    assert at.session_state["retrieved_user_with_scores"] is not None
    assert len(at.session_state["retrieved_user_with_scores"].scores) >= 3

    # Test fetching non-existent user
    at.text_input[1].set_value("non_existent_id").run()
    at.button[2].click().run()
    assert "User not found." in at.warning[0].value

def test_caching_with_redis(at):
    """
    Test Redis caching functionality.
    Requires DB to be initialized first.
    """
    # Initialize DB first
    at.sidebar.selectbox[0].set_value("2. DB Connectivity & Pooling").run()
    at.button[0].click().run()
    assert at.session_state["db_initialized"]

    # Navigate to Caching with Redis page
    at.sidebar.selectbox[0].set_value("4. Caching with Redis").run()

    # The app code itself creates the 'cache_demo@innovateai.com' user
    # and a score if they don't exist. So we just need to run it once.
    # We should see the info message indicating the user is ready.
    at.run() # Rerun to ensure the user setup logic completes
    assert at.session_state["user_for_caching_id"] is not None
    assert f"User for caching demo (ID: `{at.session_state['user_for_caching_id']}`) ready." in at.info[0].value

    user_id_for_cache = at.session_state["user_for_caching_id"]

    # Test Fetch User (Cached) - first call should be a miss, second a hit
    # First call:
    at.button[0].click().run()
    assert "Cached User Details:" in at.write[0].value
    assert "Time taken:" in at.markdown[2].value
    assert at.session_state["cache_misses"] == 1 # Mock client, so this is based on app's heuristic

    # Second call (should be a hit for mock logic, but app's counter is simpler)
    at.button[0].click().run()
    assert at.session_state["cache_hits"] == 1 # If the mock returns a value on 'get'

    # Test Fetch Latest AIRScore (Cached)
    at.button[1].click().run()
    assert "Cached Latest AIRScore Details:" in at.write[1].value
    assert "Time taken:" in at.markdown[3].value

    # Test Invalidate Cache
    at.button[2].click().run()
    assert f"Cache invalidated for user ID: {user_id_for_cache}" in at.success[0].value

    # Check cache metrics display
    assert f"Mock Cache Hits: {at.session_state['cache_hits']}" in at.markdown[4].value
    assert f"Mock Cache Misses: {at.session_state['cache_misses']}" in at.markdown[5].value

def test_eventing_outbox_pattern(at):
    """
    Test the Outbox Pattern eventing system.
    Requires DB to be initialized first.
    """
    # Initialize DB first
    at.sidebar.selectbox[0].set_value("2. DB Connectivity & Pooling").run()
    at.button[0].click().run()
    assert at.session_state["db_initialized"]

    # Navigate to Eventing (Outbox Pattern) page
    at.sidebar.selectbox[0].set_value("5. Eventing (Outbox Pattern)").run()

    # The app code itself creates the 'event_demo@innovateai.com' user
    # if it doesn't exist. Run once to ensure that logic completes.
    at.run()
    assert at.session_state["event_user_id"] is not None
    assert f"User for eventing demo (ID: `{at.session_state['event_user_id']}`) ready." in at.info[0].value

    user_id_for_event = at.session_state["event_user_id"]

    # Generate an AIRScore and a Pending Domain Event
    at.button[0].click().run() # Calculate & Store AIRScore
    assert "AIRScore (ID:" in at.success[0].value
    assert "DomainEvent (ID:" in at.success[0].value
    assert "recorded as 'pending'." in at.success[0].value

    # Check that there's a pending event displayed
    at.button[3].click().run() # Refresh Event Status Now
    assert len(at.session_state["pending_events_display"]) > 0
    assert at.session_state["pending_events_display"][0]["Status"] == "pending"
    assert "No published events yet." in at.info[1].value

    # Run Event Publisher (Simulate Background)
    # This will trigger the publisher simulation.
    at.button[1].click().run()
    assert at.session_state["event_publisher_running"] == True
    # The progress bar and status text will update during the run.
    # We can check the final state after the simulation completes.
    at.run() # Rerun to reflect the final state after the publisher finishes
    assert at.session_state["event_publisher_running"] == False
    assert "Publisher simulation finished." in at.empty[0].value # Check for the final status message
    assert "Processed 1 events." in at.success[1].value # Assuming one event was created and processed

    # Refresh Event Status Now to see the processed event
    at.button[3].click().run()
    assert "No pending events." in at.info[0].value
    assert len(at.session_state["processed_events_display"]) > 0
    assert at.session_state["processed_events_display"][0]["Status"] == "published"
