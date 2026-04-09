import sqlite3

conn = sqlite3.connect("sales.db")
cursor = conn.cursor()

# Create tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS customers (
    id INTEGER PRIMARY KEY,
    name TEXT,
    region TEXT,
    signup_date TEXT
)""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER,
    amount REAL,
    product TEXT,
    order_date TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(id)
)""")

# Insert sample data
customers = [
    (1, "Rajesh Kumar", "north", "2024-01-15"),
    (2, "Priya Sharma", "south", "2024-03-20"),
    (3, "Amit Patel", "north", "2024-06-10"),
    (4, "Sneha Gupta", "west", "2024-02-28"),
    (5, "Vikram Singh", "east", "2024-07-05"),
    (6, "Ananya Reddy", "south", "2024-04-12"),
    (7, "Rohit Verma", "north", "2024-08-18"),
    (8, "Kavita Joshi", "west", "2024-05-22"),
    (9, "Deepak Mishra", "east", "2024-09-30"),
    (10, "Neha Agarwal", "south", "2024-11-14"),
]

orders = [
    (1, 1, 15000, "Dashboard Pro", "2025-01-10"),
    (2, 1, 22000, "Analytics Suite", "2025-03-15"),
    (3, 2, 8500, "Dashboard Pro", "2025-02-20"),
    (4, 3, 45000, "Enterprise Plan", "2025-01-25"),
    (5, 3, 12000, "Dashboard Pro", "2025-04-10"),
    (6, 4, 35000, "Analytics Suite", "2025-02-14"),
    (7, 5, 9500, "Dashboard Pro", "2025-03-22"),
    (8, 6, 18000, "Analytics Suite", "2025-05-08"),
    (9, 7, 52000, "Enterprise Plan", "2025-04-30"),
    (10, 7, 15000, "Dashboard Pro", "2025-06-12"),
    (11, 8, 28000, "Analytics Suite", "2025-03-18"),
    (12, 9, 11000, "Dashboard Pro", "2025-05-25"),
    (13, 10, 7500, "Dashboard Pro", "2025-06-30"),
    (14, 2, 32000, "Enterprise Plan", "2025-07-15"),
    (15, 4, 19000, "Dashboard Pro", "2025-07-20"),
    (16, 5, 41000, "Enterprise Plan", "2025-08-10"),
    (17, 6, 14000, "Dashboard Pro", "2025-08-22"),
    (18, 8, 38000, "Enterprise Plan", "2025-09-05"),
    (19, 1, 27000, "Analytics Suite", "2025-09-18"),
    (20, 3, 33000, "Analytics Suite", "2025-10-02"),
]

cursor.executemany("INSERT OR REPLACE INTO customers VALUES (?,?,?,?)", customers)
cursor.executemany("INSERT OR REPLACE INTO orders VALUES (?,?,?,?,?)", orders)

conn.commit()
conn.close()

print("Database created: sales.db")
print(f"  Customers: {len(customers)}")
print(f"  Orders: {len(orders)}")