from ast import List
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
from datetime import timedelta
from app.config import config



def create_indexes():
    try:
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')  # Quick connection test
        print("✅ Connected to MongoDB.")

        db = client[config.MONGO_DB_NAME]

        # === 1. Indexes for individual backtest run logs ===
        strategy_runs = db[config.MONGO_COLLECTION_BACKTEST_RESULTS]

        print("📌 Creating indexes on 'strategy_runs' collection...")
        strategy_runs.create_index([
            ("session", 1),
            ("is_expiry", 1),
            ("performance_score", -1)
        ])
        strategy_runs.create_index([
            ("market_condition", 1),
            ("timeframe", 1)
        ])
        strategy_runs.create_index("logged_at", expireAfterSeconds=60 * 24 * 3600)  # TTL index for 60 days
        print("✅ Indexes created on 'strategy_runs'.")

        # === 2. Indexes for tuned params (best strategy params per context) ===
        strategy_params = db[config.MONGO_COLLECTION_TUNED_PARAMS]

        print("📌 Creating indexes on 'strategy_best_params' collection...")
        strategy_params.create_index([
            ("strategy", 1),
            ("timeframe", 1),
            ("symbol", 1),
            ("day", 1),
            ("session", 1),
            ("is_expiry", 1)
        ], unique=True)
        print("✅ Indexes created on 'strategy_best_params'.")

        print("🎉 All indexes created successfully!")

    except ConnectionFailure:
        print(f"❌ MongoDB connection failed at {config.MONGO_URI}")
    except OperationFailure as op_e:
        print(f"❌ MongoDB operation failed: {op_e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        try:
            client.close()
            print("🔒 MongoDB connection closed.")
        except:
            pass
def drop_all_indexes():
    try:
        client = MongoClient(config.MONGO_URI, serverSelectionTimeoutMS=5000)
        client.admin.command('ismaster')  # Connection test
        print("✅ Connected to MongoDB.")

        db = client[config.MONGO_DB_NAME]

        collections_to_clean = [
            config.MONGO_COLLECTION_BACKTEST_RESULTS,
            config.MONGO_COLLECTION_TUNED_PARAMS
        ]

        for collection_name in collections_to_clean:
            collection = db[collection_name]
            indexes = collection.index_information()
            print(f"\n🔍 Indexes for '{collection_name}':")
            for name in indexes:
                if name != "_id_":
                    print(f"🗑️ Dropping index: {name}")
                    collection.drop_index(name)
            print(f"✅ All non-_id indexes dropped for '{collection_name}'.")

        print("\n🎉 Index cleanup complete.")

    except ConnectionFailure:
        print(f"❌ MongoDB connection failed at {config.MONGO_URI}")
    except OperationFailure as op_e:
        print(f"❌ MongoDB operation failed: {op_e}")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
    finally:
        try:
            client.close()
            print("🔒 MongoDB connection closed.")
        except:
            pass

if __name__ == "__main__":
    drop_all_indexes()
    create_indexes()
