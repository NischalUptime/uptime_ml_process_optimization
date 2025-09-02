import psycopg2
import structlog
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = structlog.get_logger(__name__)

class DatabaseManager:
    def __init__(self, configuration: Dict = None):
        """Initialize DatabaseManager with configuration from config.yaml.
        
        Args:
            configuration: Configuration dictionary containing database settings
        """
        if configuration is None:
            logger.error("Database configuration not provided")
            raise ValueError("Configuration must be provided")
        
        self.db_config = configuration.get('database', {})
        if not self.db_config:
            logger.error("Database configuration section missing from config.yaml")
            raise ValueError("Database configuration not found in config")
        
        # Log configuration details (without sensitive info)
        logger.info("Initializing database manager", 
                   host=self.db_config.get('host', 'unknown'),
                   port=self.db_config.get('port', 'unknown'),
                   dbname=self.db_config.get('dbname', 'unknown'),
                   user=self.db_config.get('user', 'unknown'))
            
        self.conn = None
        self.cursor = None

    def connect(self):
        """Establish database connection"""
        if not self.conn:
            try:
                logger.info("Establishing database connection", 
                           host=self.db_config.get('host'),
                           dbname=self.db_config.get('dbname'))
                self.conn = psycopg2.connect(**self.db_config)
                self.cursor = self.conn.cursor()
                logger.info("Database connection established successfully")
            except psycopg2.Error as e:
                logger.error("Failed to connect to database", 
                            error=str(e),
                            host=self.db_config.get('host'),
                            dbname=self.db_config.get('dbname'))
                raise
            except Exception as e:
                logger.error("Unexpected error during database connection", error=str(e))
                raise

    def disconnect(self):
        """Close database connection"""
        try:
            if self.cursor:
                self.cursor.close()
                logger.debug("Database cursor closed")
            if self.conn:
                self.conn.close()
                logger.debug("Database connection closed")
                self.conn = None
                self.cursor = None
        except Exception as e:
            logger.warning("Error during database disconnect", error=str(e))

    def get_latest_data(self, required_vars: List[str], last_timestamp: Optional[datetime] = None, min_lag: Optional[int] = None, max_lag: Optional[int] = None) -> Dict[str, List[Dict]]:
        """
            Fetch process data from the database using one of three strategies:
            1. Lag window (min_lag and max_lag provided)
            2. Rows after a given last_timestamp
            3. Latest single row (default)

            Returns:
            dict: {"rows": [{"timestamp": ..., "data": {...}}, ...]}
        """
        logger.info(
            "Fetching data from database",
            required_vars_count=len(required_vars),
            last_timestamp=last_timestamp,
            min_lag=min_lag,
            max_lag=max_lag,
        )

        try:
            self.connect()
            # Format column names with proper quotes
            columns = ', '.join(f'"{var}"' for var in required_vars)


            # -------------------
            # Case 1: lag window
            # -------------------
            if min_lag is not None and max_lag is not None:
                now = datetime.now()
                start_time = now - timedelta(minutes=max_lag)
                end_time = now - timedelta(minutes=min_lag)

                query = f"""
                SELECT "timestamp", {columns}
                FROM process_data
                WHERE "timestamp" BETWEEN %s AND %s
                ORDER BY "timestamp" ASC;
                """
                params = (start_time, end_time)
                logger.debug("Executing lag-window query", 
                            query=query.strip(),
                            start_time=start_time,
                            end_time=end_time)
            
            # -------------------
            # Case 2: last_timestamp present
            # -------------------
            elif last_timestamp:
                query = f"""
                SELECT "timestamp", {columns}
                FROM process_data
                WHERE "timestamp" > %s
                ORDER BY "timestamp" DESC
                LIMIT 1;
                """
                params = (last_timestamp,)
                logger.debug("Executing timestamped query",
                           query=query.strip(),
                           last_timestamp=last_timestamp)

            # -------------------
            # Case 3: latest row
            # -------------------
            else:
                # If no timestamp, get the latest row
                query = f"""
                SELECT "timestamp", {columns}
                FROM process_data
                ORDER BY "timestamp" DESC
                LIMIT 1;
                """
                params = ()
                logger.debug("Executing latest data query", query=query.strip())
            
            self.cursor.execute(query, params)
            # Get column names and fetch data
            column_names = [desc[0] for desc in self.cursor.description]
            raw_rows = self.cursor.fetchall()

            if not raw_rows:
                logger.warning("No data found in database")
                raise Exception("No data found in database")
                
            records = []

            for row in raw_rows:
                row_dict = dict(zip(column_names, row))
                timestamp = row_dict.pop('timestamp')  # Remove and get timestamp
                records.append({'timestamp': timestamp, 'data': row_dict})

            logger.info("Successfully fetched data from database",
                    rows_count=len(records),
                    variables=list(records[0]["data"].keys()) if records else [])

            return { "rows": records }

        except psycopg2.Error as e:
            logger.error("PostgreSQL database error", 
                        error=str(e),
                        query=query.strip() if 'query' in locals() else 'unknown')
            raise Exception(f"Database error: {str(e)}")
        except Exception as e:
            logger.error("Unexpected error fetching data", error=str(e))
            raise Exception(f"Database error: {str(e)}")
        finally:
            self.disconnect()

    def __enter__(self):
        """Context manager entry"""
        logger.debug("Entering database context manager")
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        if exc_type:
            logger.error("Exception occurred in database context", 
                        exception_type=exc_type.__name__ if exc_type else None,
                        exception_value=str(exc_val) if exc_val else None)
        logger.debug("Exiting database context manager")
        self.disconnect()