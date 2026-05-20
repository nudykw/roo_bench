<!--
  Prompts for Roo Bench benchmark system
  Contains independent prompts and chains for different modes
  Generated from prompts.jsonc: python roo_bench.py --generate-md
-->

# independent
<!--
  Independent prompts - each runs without context from other modes
-->

<!--
  Architect mode prompts - test model's ability to design systems
-->
## architect

### arch_cache_system
**Name:** Distributed Cache System
**Prompt:** Create a detailed plan for a distributed caching system with: LRU eviction policy, TTL (time-to-live) invalidation, replication between nodes, eventual consistency, REST API for cache management. Describe: 1) Data structure for cache storage, 2) API endpoints with example requests, 3) Replication strategy and consistency, 4) Invalidation mechanism (by TTL and by event), 5) Node failure handling.

Additional Constraints: Consider security implications (authentication for cache API, encryption at rest). Describe sharding strategy for horizontal scaling. Address cache stampede prevention (mutex per key or probabilistic early expiration).

### arch_event_driven
**Name:** Event-Driven E-Commerce
**Prompt:** Design an event-driven system for order processing in an e-commerce platform. The system must include: event queues (RabbitMQ/Kafka-like), handlers for different event types, retry mechanism with exponential backoff, dead letter queue for failed processing, monitoring and alerting. Describe: 1) Event flow diagram, 2) Queue and topic schema, 3) Event contract with example JSON, 4) Duplicate event handling (idempotency), 5) Handler scaling strategy.

Additional Constraints: Describe exactly-once vs at-least-once delivery guarantees and their implementation. Address schema evolution strategy for events (backward/forward compatibility). Describe monitoring metrics for queue depth, processing latency, and error rates.

### arch_auth_system
**Name:** Authentication System
**Prompt:** Develop a plan for an authentication and authorization system with: OAuth 2.0 / OpenID Connect, JWT tokens with access and refresh mechanism, role-based access control (RBAC), multi-factor authentication (MFA), rate limiting for API. Describe: 1) Authorization flow (Authorization Code with PKCE), 2) JWT payload structure, 3) Token refresh mechanism with rotation, 4) Role model and access matrix, 5) Secret storage strategy.

Additional Constraints: Address token revocation strategy (blocklist vs short expiry). Describe protection against replay attacks, CSRF, and brute-force login attempts. Consider multi-tenancy support with isolated token scopes.

<!--
  Code mode prompts - test model's ability to implement code
-->
## code

### code_thread_pool
**Name:** ThreadPoolExecutor from Scratch
**Prompt:** Implement a ThreadPoolExecutor class in Python from scratch (without using standard concurrent.futures). Requirements: Pool of N threads (configurable), task queue (queue.Queue), support for submit() and map() methods, Future objects for result retrieval, graceful shutdown with waiting for task completion, exception handling in tasks, context manager (__enter__/__exit__). Add: type hints, docstrings for each method, usage example in __main__.

Additional Constraints: Implement thread-safe task cancellation. Add metrics tracking (tasks submitted, completed, failed, average execution time). Handle thread exceptions without crashing the pool. Implement work-stealing for load balancing when threads are idle.

### code_async_http
**Name:** Async HTTP Client
**Prompt:** Write an asynchronous HTTP client in Python using asyncio and aiohttp. Functionality: GET, POST, PUT, DELETE methods, automatic retry with exponential backoff (configurable number of attempts), timeout for connection and read, connection pooling, middleware pipeline for request/response processing, request logging, support for headers and JSON body. Add: type hints, error handling (custom exception hierarchy), usage examples for each method.

Additional Constraints: Implement circuit breaker pattern for failing endpoints. Add request/response size limits to prevent memory exhaustion. Support HTTP/2 multiplexing. Implement idempotency keys for safe retry of POST requests.

### code_pb_serializer
**Name:** Protocol Buffers Serializer
**Prompt:** Implement a serializer/deserializer in Python compatible with Protocol Buffers concept. Requirements: Field class for defining fields with types (int, string, bytes, nested message), Message class as base for all messages, serialize() and deserialize() methods with binary encoding, support for required and optional fields, support for repeated (array) fields, support for nested messages, type validation during serialization. Add: example message definition, serialization/deserialization example, error handling for invalid data.

Additional Constraints: Implement field numbering compatible with standard Protobuf wire format. Support oneof fields for mutually exclusive options. Add schema validation with detailed error messages indicating field path. Implement streaming serialization for large messages.

<!--
  Debug mode prompts - test model's ability to find and fix bugs
-->
## debug

### debug_memory_leak
**Name:** Memory Leak Detection
**Prompt:** Analyze this code for memory leaks and circular references:

```python
import gc

class Node:
    def __init__(self, value):
        self.value = value
        self.children = []
        self.parent = None
        
    def add_child(self, child):
        self.children.append(child)
        child.parent = self  # Circular reference!

class Cache:
    def __init__(self):
        self.items = {}
        self._weak_refs = []
        
    def add(self, key, obj):
        self.items[key] = obj
        # Forgot to add to _weak_refs!
        
    def get(self, key):
        return self.items.get(key)  # Never removes expired entries
        
    def cleanup(self):
        # Incomplete cleanup
        self.items.clear()

# Usage
root = Node("root")
for i in range(100):
    child = Node(f"child_{i}")
    root.add_child(child)

cache = Cache()
for i in range(1000):
    cache.add(f"item_{i}", Node(f"node_{i}"))

# root and cache not explicitly cleaned up
del root
del cache
# gc.collect() not called
```

Find: 1) All circular references, 2) Memory leaks (including unbounded cache growth), 3) Provide fixed code, 4) Explain how to prevent such problems.

### debug_race_condition
**Name:** Race Condition Detection
**Prompt:** Find race conditions in this multithreaded code and provide a fix:

```python
import threading
import time

class Counter:
    def __init__(self):
        self.value = 0
        
    def increment(self):
        temp = self.value
        time.sleep(0.001)  # Simulate work
        self.value = temp + 1
        
    def get_value(self):
        return self.value

class BankAccount:
    def __init__(self, balance):
        self.balance = balance
        self._lock = threading.Lock()
        
    def withdraw(self, amount):
        if self.balance >= amount:
            time.sleep(0.001)  # Simulate processing
            self.balance -= amount
            return True
        return False

    def deposit(self, amount):
        time.sleep(0.001)
        self.balance += amount

    def transfer(self, other, amount):
        # Deadlock potential: self.lock -> other.lock vs other.lock -> self.lock
        if self.balance >= amount:
            self.balance -= amount
            other.balance += amount

# Usage
counter = Counter()
threads = []
for _ in range(100):
    t = threading.Thread(target=counter.increment)
    threads.append(t)
    t.start()
for t in threads:
    t.join()
print(f"Counter: {counter.get_value()}")  # Expected: 100

account = BankAccount(1000)
threads = []
for _ in range(10):
    t = threading.Thread(target=account.withdraw, args=(100,))
    threads.append(t)
    t.start()
for t in threads:
    t.join()
print(f"Balance: {account.balance}")  # Can be negative!
```

Find: 1) All race conditions (including deadlock potential in transfer), 2) Explain the mechanism of occurrence, 3) Provide fix using lock/RLock/atomic operations.

### debug_async_issues
**Name:** Async Debugging
**Prompt:** Debug this asynchronous code, find all problems:

```python
import asyncio
import aiohttp

async def fetch_data(session, url):
    async with session.get(url) as response:
        data = await response.json()
        return data

async def fetch_all(urls):
    results = []
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_data(session, url) for url in urls]
        # Problem: no timeout
        data = await asyncio.gather(*tasks)
        results.extend(data)
    return results

async def process_with_retry(url, retries=3):
    for i in range(retries):
        try:
            async with aiohttp.ClientSession() as session:
                data = await fetch_data(session, url)
                return data
        except Exception:
            if i == retries - 1:
                raise
            # Problem: no delay between retries
            pass

async def cancel_task(task):
    task.cancel()
    # Missing: await task to properly handle CancelledError
    # This leaves the task in a cancelled but unawaited state

async def main():
    urls = [
        "https://api.example.com/data1",
        "https://api.example.com/data2",
        "https://api.example.com/data3",
    ]
    
    # Problem: unlimited parallel requests
    results = await fetch_all(urls)
    
    # Problem: incomplete error handling
    for url in urls:
        try:
            data = await process_with_retry(url)
            print(data)
        except Exception:
            print(f"Failed: {url}")

asyncio.run(main())
```

Find and fix: 1) Missing timeouts, 2) No delay between retries, 3) Unlimited parallelism, 4) Unawaited cancelled tasks, 5) Other problems.

<!--
  Chain prompts - context flows between modes
-->
# chains

### chain_rest_api
**Name:** REST API Server
**Description:** Full lifecycle: design REST API -> implement it -> debug the implementation
**Prompts:**
- **architect:** Create a detailed plan for a REST API server on Python using asyncio. Requirements: RESTful endpoints - GET/POST/PUT/DELETE for /users and /items resources. Middleware system for: Authentication (JWT), Rate limiting (100 req/min per IP), Request/Response logging, CORS handling. Health check endpoint (/health). In-memory storage with ability to replace with database. Graceful shutdown waiting for request completion. Describe: 1) Project structure (files and modules), 2) Middleware pipeline diagram, 3) JWT token format and authentication flow, 4) Rate limiting strategy (token bucket algorithm), 5) Error handling (custom exception hierarchy).

Additional Requirements: Implement request tracing with unique correlation IDs propagated through middleware. Add support for API versioning (/v1/, /v2/). Describe database migration strategy for schema changes. Address DDoS protection at the middleware level.
- **code:** Implement a REST API server based on the following architect plan: [ARCHITECT_PLAN]. Requirements: Use asyncio and aiohttp, implement all described middleware, include JWT authentication, implement rate limiting via token bucket, add health check, implement graceful shutdown, add type hints and docstrings, write curl examples for each endpoint.

Additional Requirements: Implement structured logging with request ID correlation. Add request validation using pydantic models. Implement health check with database connectivity verification. Add metrics endpoint exposing request counts, latency percentiles, and error rates.
- **debug:** Analyze and fix problems in this REST API server: [CODE_FROM_CODE]. Check for: 1) Race conditions with in-memory storage, 2) Resource leaks (unclosed connections, sessions), 3) Rate limiting logic errors, 4) JWT validation issues, 5) Graceful shutdown correctness, 6) Edge case handling.

Additional Checks: Verify proper cleanup of async resources on shutdown. Check for SQL injection vulnerabilities in query construction. Validate that rate limiting works correctly under concurrent requests. Check for memory leaks in middleware state.

### chain_task_queue
**Name:** Task Queue System
**Description:** Full lifecycle: design task queue -> implement it -> debug the implementation
**Prompts:**
- **architect:** Design a task management system (task queue) with worker pool. Requirements: Priority queues (high, medium, low), worker pool with configurable worker count, retry mechanism with exponential backoff, dead letter queue for failed tasks, monitoring: task status, worker metrics, REST API for management: add tasks, get status, cancel tasks, view metrics. Describe: 1) Task structure (Task model), 2) Task distribution algorithm to workers, 3) Retry mechanism with backoff, 4) Monitoring schema, 5) Task processing flow from creation to completion.

Additional Requirements: Describe task persistence strategy (disk-backed queue for crash recovery). Address worker failure detection and task reassignment. Design idempotency mechanism to prevent duplicate task execution.
- **code:** Implement a task management system based on the following architect plan: [ARCHITECT_PLAN]. Requirements: Priority queues (heapq for each priority level), worker pool using threading, retry with exponential backoff (min=1s, max=300s, factor=2), dead letter queue, REST API via aiohttp, metrics: active, pending, failed, completed count, type hints and docstrings.

Additional Requirements: Implement task serialization for persistence. Add worker heartbeat mechanism for failure detection. Implement task priority preemption (high priority tasks can interrupt low priority ones).
- **debug:** Debug this task management system: [CODE_FROM_CODE]. Check for: 1) Priority queue correctness, 2) Race conditions with shared resources, 3) Exponential backoff correctness, 4) Edge case handling (empty queue, all workers busy), 5) Memory leaks during long operation, 6) Task cancellation correctness.

Additional Checks: Verify task ordering guarantees within priority levels. Check for race conditions in worker assignment. Validate exponential backoff calculation correctness. Check for memory leaks in completed task storage.

### chain_websocket_chat
**Name:** WebSocket Chat
**Description:** Full lifecycle: design WebSocket chat -> implement it -> debug the implementation
**Prompts:**
- **architect:** Design a real-time chat system via WebSocket. Requirements: User connect/disconnect, create/join/leave rooms, send/receive messages, typing indicator, message history (last 100 per room), online user status, rate limiting for messages (10 seconds between messages). Describe: 1) WebSocket protocol (JSON schema), 2) Data structure for rooms and messages, 3) User connection flow, 4) Message broadcast mechanism, 5) History storage strategy.

Additional Requirements: Design message ordering guarantees across multiple servers. Address message delivery confirmation (ACK/NACK). Describe strategy for handling offline messages and catch-up on reconnect.
- **code:** Implement a WebSocket chat server based on the following architect plan: [ARCHITECT_PLAN]. Requirements: aiohttp WebSocket server, room management (dict of sets for users), in-memory history storage (deque), typing indicator via separate event, rate limiting, JSON protocol for messages.

Additional Requirements: Implement message acknowledgment system. Add reconnection logic with message catch-up. Implement rate limiting per user with sliding window algorithm.
- **debug:** Debug this WebSocket chat server: [CODE_FROM_CODE]. Check for: 1) WebSocket connection leaks, 2) Race conditions during broadcast, 3) Cleanup correctness on disconnect, 4) Memory leak in history storage, 5) Binary data handling, 6) Graceful shutdown.

Additional Checks: Verify proper cleanup on abnormal disconnect. Check for race conditions in message broadcast. Validate rate limiting accuracy under high message volume. Check for memory leaks in message history.
